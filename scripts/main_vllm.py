import asyncio
import logging
import os
import pickle
import shlex
import subprocess
import time
import urllib.request

import hydra
from hydra.core.hydra_config import HydraConfig


def _wait_http_ok(url: str, timeout_s: int = 120):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def _maybe_start_vllm(cfg):
    """Start local vLLM if configured. Return proc or None."""
    vllm_cfg = None
    if hasattr(cfg, "launcher") and hasattr(cfg.launcher, "vllm"):
        vllm_cfg = cfg.launcher.vllm

    if not vllm_cfg or not getattr(vllm_cfg, "enable", False):
        return None

    host = getattr(vllm_cfg, "host", "127.0.0.1")
    port = int(getattr(vllm_cfg, "port", 8000))
    model = vllm_cfg.model

    cmd = ["vllm", "serve", model, "--host", host, "--port", str(port), "--dtype", "auto"]

    served_name = getattr(vllm_cfg, "served_model_name", None)
    if served_name:
        cmd += ["--served-model-name", served_name]

    tool_parser = getattr(vllm_cfg, "tool_call_parser", None)
    if tool_parser:
        cmd += ["--tool-call-parser", tool_parser]

    if getattr(vllm_cfg, "enable_lora", False):
        cmd += ["--enable-lora"]
        for name, repo in (getattr(vllm_cfg, "lora_modules", {}) or {}).items():
            cmd += ["--lora-modules", f"{name}={repo}"]

    extra_args = getattr(vllm_cfg, "extra_args", "")
    if extra_args:
        cmd += shlex.split(extra_args)

    logging.info(f"Starting vLLM with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    # Wait for vLLM to be ready
    _wait_http_ok(f"http://{host}:{port}/v1/models", timeout_s=900)

    # ✅ MUST include '/v1'
    base = f"http://{host}:{port}/v1"
    os.environ["HOSTED_VLLM_API_BASE"] = base
    os.environ.setdefault("HOSTED_VLLM_API_KEY", "")
    # Safety: ensure OpenAI client also targets local vLLM if used under the hood
    os.environ["OPENAI_BASE_URL"] = base

    logging.info(f"vLLM started successfully at {base}")
    return proc


async def chunked(aiter, size):
    batch = []
    async for item in aiter:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def run_episode(env, agents):
    observations = await env.reset()
    while not env.done():
        futures = {agent_id: agents[agent_id].step(observation) for agent_id, observation in observations.items()}
        responses = dict(zip(futures.keys(), await asyncio.gather(*futures.values())))
        logging.info(f"response: {responses}")
        observations = await env.step(responses)
    return env.get_result()


@hydra.main(version_base=None, config_path="../configs", config_name="config_vllm")
def main(cfg):
    vllm_proc = _maybe_start_vllm(cfg)
    try:

        async def run():
            task = hydra.utils.instantiate(cfg.task)
            batchsize = cfg.batchsize

            for i in range(cfg.num_episodes):
                results = []

                async for env_batch in chunked(task.iterate_environments(), batchsize):
                    tasks = []
                    for env in env_batch:
                        cfg.agent.num_agents = env.num_agents()
                        agents = hydra.utils.instantiate(cfg.agent)
                        # Update agent configs if provided
                        configs = env.get_default_agent_configs()
                        if configs is not None:
                            for agent_id, agent in enumerate(agents):
                                if hasattr(agent, "update_config"):
                                    agent.update_config(configs[agent_id])

                        tasks.append(run_episode(env, agents))

                    results.extend(await asyncio.gather(*tasks))

                aggregated_results = task.aggregate_results(results)
                run_dir = HydraConfig.get().run.dir
                out = os.path.join(run_dir, f"aggregated_results_episode_{i}.pkl")
                with open(out, "wb") as f:
                    pickle.dump(aggregated_results, f)
                logging.info(f"Aggregated results saved to: {out}")

        asyncio.run(run())
    finally:
        if vllm_proc is not None:
            logging.info("Shutting down vLLM...")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
                logging.info("vLLM shut down successfully")
            except subprocess.TimeoutExpired:
                logging.warning("vLLM did not terminate gracefully, forcing kill...")
                vllm_proc.kill()


if __name__ == "__main__":
    main()
