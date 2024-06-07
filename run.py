import regelum as rg
import torch

@rg.main(config_path="presets", config_name="main_pretrain")
def launch_pretrain(cfg):
    scenario = ~cfg.scenario  # instantiate the scenario from config
    scenario.run()  # run it
    torch.save(scenario.portfolio_critic.model.state_dict(), './portfolio_critic.pt')
    torch.save(scenario.policy.portfolio_model.state_dict(), './portfolio_actor.pt')
    torch.save(scenario.market_critic.model.state_dict(), './market_critic.pt')
    torch.save(scenario.policy.market_model.state_dict(), './market_actor.pt')


@rg.main(config_path="presets", config_name="main_train")
def launch_train(cfg):
    scenario = ~cfg.scenario  # instantiate the scenario from config
    scenario.run()  # run it
    torch.save(scenario.portfolio_critic.model.state_dict(), './portfolio_critic.pt')
    torch.save(scenario.policy.portfolio_model.state_dict(), './portfolio_actor.pt')
    torch.save(scenario.market_critic.model.state_dict(), './market_critic.pt')
    torch.save(scenario.policy.market_model.state_dict(), './market_actor.pt')

if __name__ == "__main__":
    launch_pretrain()
    launch_train()
    pass
