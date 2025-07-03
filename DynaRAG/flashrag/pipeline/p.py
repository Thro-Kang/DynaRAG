from flashrag.pipeline import BasicPipeline
from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate


class ToyPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        # Load your own components
        super().__init__(config, prompt_template)
        self.retriever = get_retriever(config)
        self.refiner = None
        self.generator = get_generator(config)


    def run(self, dataset, do_eval=True):
        # Complete your own process logic

        # get attribute in dataset using `.`
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)
        # use `update_output` to save intermeidate data
        dataset.update_output("retrieval_results", retrieval_results)
        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)
        return dataset
