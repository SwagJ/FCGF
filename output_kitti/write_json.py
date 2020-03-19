import json
from config import get_config



config = get_config()

print(config)

#json.dump(
#        config,
#        open('config.json', 'w'),
#        indent=4,
#        sort_keys=False)
