from langchain_core.runnables import RunnableLambda

class OCRNERPipeline:
    def __init__(self, ocr_model, ner_model):
        self.pipeline = (
            RunnableLambda(lambda x: ocr_model.extract_text(x))
            | RunnableLambda(
                lambda text: {
                    "text": text,
                    "entities": ner_model.extract_entities(text)
                }
            )
        )

    def run(self, sample):
        return self.pipeline.invoke(sample)
