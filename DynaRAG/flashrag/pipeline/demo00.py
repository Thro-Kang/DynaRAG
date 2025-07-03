from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import Demo01Pipeline
from flashrag.prompt import PromptTemplate

config = Config(config_file_path="/root/autodl-tmp/FlashRAG/flashrag/config/basic_config.yaml")

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = Demo01Pipeline(config)
output_dataset = pipeline.run(test_data, do_eval=True)
print("---generation output---")
print(output_dataset.pred)
