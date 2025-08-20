def create_env(env_name, tokenizer):
    env_name = env_name.lower()
    if env_name == 'nco':
        from cecl.envs.nco_env import NCOEnv
        env = NCOEnv(tokenizer)
    elif env_name == 'nco_enhanced':
        from cecl.envs.nco_env_enhanced import EnhancedNCOEnv
        env = EnhancedNCOEnv(tokenizer)
    elif env_name == 'nco_single':
        from cecl.envs.nco_env_single import SingleQuarterNCOEnv
        env = SingleQuarterNCOEnv(tokenizer)
    else:
        raise ValueError(f"Unknown environment name: {env_name}. Available: nco, nco_enhanced, nco_single")
    return env