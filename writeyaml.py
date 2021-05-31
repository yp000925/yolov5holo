import yaml


c ={}
c['names'] = [str(x) for x in range(1,257)]
c['train'] = '../data/images/train/'
c['val'] = '../data/images/validation/'
c['nc'] = 256


with open(r'test.yaml', 'w') as file:
    documents = yaml.dump(c, file, default_flow_style=False)


