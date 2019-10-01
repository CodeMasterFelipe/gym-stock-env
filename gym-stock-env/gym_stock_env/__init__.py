from gym.envs.registration import register

register(
    id='stock_env-v0',
    entry_point='gym_stock_env.envs:StockMarketEnv_Train',
)
