from sacred import Experiment

ex = Experiment('hello_config')

# configuration
@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

# main script that will run automatically
@ex.automain
def my_main(message):
    print(message)