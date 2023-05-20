from configuration import get_config
from algorithm import trainer_select

if __name__ == '__main__':
    config = get_config()
    print(config)
    trainer = trainer_select(config,**config)
    trainer.train()