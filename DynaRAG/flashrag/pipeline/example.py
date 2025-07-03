from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

config_dict = {
                'data_dir': '/root/autodl-tmp/FlashRAG/examples/quick_start/dataset/',
                'index_path': '/root/autodl-tmp/FlashRAG/examples/quick_start/indexes/e5_Flat.index',
                'corpus_path': '/root/autodl-tmp/FlashRAG/examples/quick_start/indexes/general_knowledge.jsonl',
                'model2path': {'e5': "/root/autodl-tmp/FlashRAG/e5-base-v2", 'llama3-8B-instruct': "/root/autodl-tmp/FlashRAG/Meta-Llama-3-8B-Instruct"},
                'generator_model': 'llama3-8B-instruct',
                'retrieval_method': 'e5',
                'metrics': ['em','f1','acc'],
                'retrieval_topk': 1,
                'save_intermediate_data': True
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
pipeline = SequentialPipeline(config)

output_dataset = pipeline.run(test_data,do_eval=True)
print("---generation output---")
print(output_dataset.pred)