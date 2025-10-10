import asyncio

import hydra

from shachi.env.oasis.snsenv import SNSTask
from shachi.env.stockagent.stockenv import StockAgentTask


async def chunked(aiter, size):
    batch = []
    async for item in aiter:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    async def run():
        for i in range(1):
            task_1 = StockAgentTask(1, 111, 264, 3, True)
            task_2 = SNSTask(1, "gpt_example.yaml", 3)
            batchsize = 1

            env_batch_1 = chunked(task_1.iterate_environments(), batchsize)
            env_batch_2 = chunked(task_2.iterate_environments(), batchsize)

            batch1 = await env_batch_1.__anext__()
            batch2 = await env_batch_2.__anext__()

            env1 = batch1[0]
            env2 = batch2[0]

            cfg.agent.num_agents = env1.num_agents()
            cfg.agent.model = "openai/gpt-3.5-turbo"
            agents = hydra.utils.instantiate(cfg.agent)

            env1_observations = await env1.reset()
            env2_observations = await env2.reset()
            while not env1.done():
                # env2
                env2_futures = {
                    agent_id: agents[agent_id].step(observation) for agent_id, observation in env2_observations.items()
                }
                env2_responses = dict(zip(env2_futures.keys(), await asyncio.gather(*env2_futures.values())))
                env2_observations = await env2.step(env2_responses)

                # env1
                for _ in range(50):
                    env1_futures = {
                        agent_id: agents[agent_id].step(observation)
                        for agent_id, observation in env1_observations.items()
                    }
                    env1_responses = dict(zip(env1_futures.keys(), await asyncio.gather(*env1_futures.values())))
                    env1_observations = await env1.step(env1_responses)

    asyncio.run(run())


if __name__ == "__main__":
    main()
