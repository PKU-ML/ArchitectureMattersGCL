"""hydra:
  job_logging:
#    formatters:
#      simple:
#        format: '[]'
    root:
      handlers: [file, console]  # logging to file only.
  run:
    #dir: logs/${dataset}
    dir: logs/SimCLR/${dataset}"""
d = {

'dataset': 'cifar10',
'data_dir': 'data',

# model
'backbone': 'resnet18', # or resnet34, resnet50
'projection_dim': 128, # "[...] to project the representation to a 128-dimensional latent space"

# train options

'batch_size': 512,
'workers': 16,
'epochs': 200,
'log_interval': 100,


# loss options
'optimizer': 'sgd', # or LARS (experimental)
'learning_rate': 0.6, # initial lr = 0.3 * batch_size / 256
'momentum': 0.9,
'weight_decay': 1.0e-6, # "optimized using LARS [...] and weight decay of 10−6"
'temperature': 0.5, # see appendix B.7.: Optimal temperature under different batch sizes


# finetune options
'finetune_epochs': 100,
'load_epoch': 1000  # checkpoint for finetune
}


class a:
    def __init__(self):
        for k in d.keys():
           setattr(self, k, d[k])
    
def get_args():
  args = a()
  
  return args