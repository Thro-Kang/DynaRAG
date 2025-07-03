from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate


class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template=None):
        # Initialize the pipeline with configuration settings.
        self.config = config
        self.device = config['device']
        self.retriever = get_retriever(config)  # Set up the retriever component.
        self.evaluator = Evaluator(config)  # Set up the evaluator component.
        self.save_retrieval_cache = config['save_retrieval_cache']

        if prompt_template is None:
            prompt_template = PromptTemplate(config)  # Set up prompt template if not provided.
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in
                              raw_pred]  # Process predictions if function is provided.
            dataset.update_output('raw_pred', raw_pred)
            dataset.update_output('pred', processed_pred)

        if do_eval:
            # Evaluate and print the results
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # Save retrieval cache if specified in config
        if self.save_retrieval_cache:
            self.retriever._save_cache()
        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """
        super().__init__(config, prompt_template)
        self.retriever = get_retriever(config)  # Initialize the retriever.

        # TODO: add rewriter module

        self.use_fid = config['use_fid']  # Determine if FiD generation is used.
        self.generator = None
        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)  # Initialize the refiner if specified.
            if 'kg' in config['refiner_name'].lower():
                self.generator = get_generator(config)  # Initialize the generator if the refiner includes 'kg'.
        else:
            self.refiner = None
            self.generator = get_generator(config)  # Initialize the generator if no refiner is specified.

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # Direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)  # Generate answers based on prompts.
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)  # Evaluate the results.
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)  # Perform batch search to retrieve documents.
        dataset.update_output('retrieval_result', retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if 'llmlingua' in self.refiner.name and input_prompt_flag:
                # Input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output('prompt', input_prompts)
                input_prompts = self.refiner.batch_run(dataset)  # Refine input prompts.
            else:
                # Input retrieval docs
                refine_results = self.refiner.batch_run(dataset)  # Refine retrieval results.
                dataset.update_output('refine_result', refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output('prompt', input_prompts)

        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )
        # Delete used refiner to release memory
        if self.refiner:
            del self.refiner
            if self.generator is None:
                self.generator = get_generator(self.config)
        pred_answer_list = self.generator.generate(input_prompts)  # Generate predictions based on prompts.
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)  # Evaluate the results.

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)  # Initialize the judger.

        self.sequential_pipeline = SequentialPipeline(config, prompt_template)  # Initialize the sequential pipeline.
        from flashrag.prompt import PromptTemplate
        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # Determine whether to use retrieval or not
        judge_result = self.judger.judge(dataset)
        dataset.update_output('judge_result', judge_result)
        # Split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)
        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)
        # Merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class Demo01Pipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        # Load your own components
        super().__init__(config, prompt_template)
        self.retriever = get_retriever(config)  # Initialize the retriever.
        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)  # Initialize the refiner if specified.
            if 'kg' in config['refiner_name'].lower():
                self.generator = get_generator(config)  # Initialize the generator if the refiner includes 'kg'.
        else:
            self.refiner = None
            self.generator = get_generator(config)  # Initialize the generator if no refiner is specified.

    def run(self, dataset, do_eval=True):
        # Complete your own process logic
        # Get attribute in dataset using `.`
        input_query = dataset.question
        retrieval_results = self.retriever.batch_search(input_query)  # Perform batch search to retrieve documents.
        # Use `update_output` to save intermediate data
        dataset.update_output("retrieval_result", retrieval_results)
        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]
        pred_answer_list = self.generator.generate(input_prompts)  # Generate answers based on prompts.
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval)  # Evaluate the results.
        return dataset


class JudgerPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        super().__init__(config, prompt_template)
        self.judger = get_judger(config)  # Initialize the judger.
        self.sequential_pipeline = SequentialPipeline(config, prompt_template)  # Initialize the sequential pipeline.
        from flashrag.prompt import PromptTemplate
        self.zero_shot_template = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # Determine whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output('judge_result', judge_result)
        # Split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)
        # Use SequentialPipeline to handle datasets that require retrieval
        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        # Use naive generation for datasets that do not require retrieval
        self.sequential_pipeline.prompt_template = self.zero_shot_template
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)
        # Merge datasets
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)
        # Evaluate results
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset
