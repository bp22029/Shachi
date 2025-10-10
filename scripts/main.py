import asyncio
import logging
import os
import pickle

import hydra
from hydra.core.hydra_config import HydraConfig


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
