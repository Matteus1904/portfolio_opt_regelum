import regelum as rg
import torch

@rg.main(config_path="presets", config_name="main")
def launch(cfg):
    scenario = ~cfg.scenario  # instantiate the scenario from config
    scenario.run()  # run it
    torch.save(scenario.portfolio_critic.model.state_dict(), './portfolio_critic.pt')
    torch.save(scenario.policy.portfolio_model.state_dict(), './portfolio_actor.pt')


if __name__ == "__main__":
    job_results = launch()
    pass
