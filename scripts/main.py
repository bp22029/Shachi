import asyncio
import logging
import os
import pickle

import hydra
import litellm
from hydra.core.hydra_config import HydraConfig

# --- ローカルLLMサーバー(MLX等)対策 ---
# サーバーがまれに壊れた(=JSONとして解析不能な)応答を返したり、リクエストが
# 長時間ハングすることがある。タイムアウトを短めにして各試行を打ち切り、
# litellm.acompletion をリトライ付きラッパーで包んで自動で取り直す。
# 各エージェントは `litellm.acompletion` をモジュール属性経由で呼ぶため、
# ここで差し替えれば全タスクに一括で適用される。
litellm.request_timeout = 150  # 秒。通常の応答は数十秒なので十分な余裕
litellm.num_retries = 2        # 接続レベルの一時的失敗向け(ラッパーとは別レイヤ)

_LLM_MAX_ATTEMPTS = 8
_orig_acompletion = litellm.acompletion


async def _acompletion_with_retry(*args, **kwargs):
    last_exc = None
    for attempt in range(1, _LLM_MAX_ATTEMPTS + 1):
        try:
            return await _orig_acompletion(*args, **kwargs)
        except Exception as e:  # 壊れた応答(InternalServerError)/タイムアウト等を再試行
            last_exc = e
            logging.warning(
                f"LLM 呼び出し失敗 (試行 {attempt}/{_LLM_MAX_ATTEMPTS}): "
                f"{type(e).__name__}: {str(e)[:150]} — リトライします"
            )
            await asyncio.sleep(min(2 ** attempt, 20))
    logging.error(f"LLM 呼び出しが {_LLM_MAX_ATTEMPTS} 回とも失敗しました")
    raise last_exc


litellm.acompletion = _acompletion_with_retry


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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
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

                    configs = env.get_default_agent_configs()
                    if configs is not None:
                        for agent_id, agent in enumerate(agents):
                            if hasattr(agent, "update_config"):
                                agent.update_config(configs[agent_id])
                    tasks.append(run_episode(env, agents))

                results.extend(await asyncio.gather(*tasks))
            aggregated_results = task.aggregate_results(results)

            run_dir = HydraConfig.get().run.dir
            filename = f"aggregated_results_episode_{i}.pkl"
            filepath = os.path.join(run_dir, filename)
            with open(filepath, "wb") as f:
                pickle.dump(aggregated_results, f)
            logging.info(f"Aggregated results saved to: {filepath}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
