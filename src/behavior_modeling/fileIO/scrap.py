def save_experiment(save_name: str, task: rnn_task.RnnMDP, agent: torch.nn.Module, performance: dict,
                    task_params: task_config.TaskParams, rnn_params: rnn_config.AgentConfig) -> None:
    exp_dict = dict(task=vars(task), task_params=vars(task_params), performance=performance, agent_params=vars(rnn_params),
               agent_name=agent.name, state_dict=task.state_dict)
    date = str(dt.date.today().isoformat())

    # save the experiment
    save_dir = data_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p_dict = save_dir / (save_name + '_dict.pkl')
    with open(p_dict, 'wb') as f:
        pkl.dump(exp_dict, f)

    # save the agent
    save_dir = model_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p_agent = save_dir / (save_name + '_agent.pkl')
    torch.save(agent.state_dict(), p_agent)

    print("[***] Experiment saved as: {}; Agent saved as: {}".format(p_dict.name, p_agent.name))


def load_experiment(exp_path: str) -> Tuple[rnn_task.RnnMDP, torch.nn.Module, pd.DataFrame, task_config.TaskParams, rnn_config.AgentConfig]:
    # p_dict = data_dir / (load_name + '_dict.pkl')
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)

    task_params = task_config.TaskParams()
    for k, v in exp['task_params'].items():
        setattr(task_params, k, v)

    rnn_params = rnn_config.AgentConfig()
    for k, v in exp['agent_params'].items():
        setattr(rnn_params, k, v)

    # agent = select_agent(exp['agent_name'], rnn_params)
    # p_agent = Path('../saved_models') / (load_name + '_agent.pkl')
    # data_dims = (1, 2, rnn_params.rnn_size)  # dX, dY, rnn_size
    # agent = pg_model.RNN_reinforce(data_dims, rnn_params)
    # agent.load_state_dict(torch.load(p_agent))
    # agent.eval()

    # for k, v in exp['task'].items():
    #     setattr(task, k, v)

    print("[***] Model restored from path: {}".format(exp_path))
    return task_params, rnn_params, exp['performance']