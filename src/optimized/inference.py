import os
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import onnxruntime as ort


class DefaultRecogniser:
    def __init__(self, model_name: str = 'raxtemur/trocr-base-ru', device: str = 'cpu'):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device=device).eval()
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.device = device

    def predict(self, images):
        pix_val = self.processor(images=images, return_tensors='pt').pixel_values.to(device=self.device)
        gen_ids = self.model.generate(pix_val, num_beams=1)
        gen_texts = self.processor.batch_decode(gen_ids, skip_special_tokens=True)

        return gen_texts


class OnnxRecogniser:
    def __init__(self, model_path: str = "../../data/models/raxtemur", device: str = 'cpu'):
        providers = [
            'CUDAExecutionProvider' if device == 'cuda:0' else 'CPUExecutionProvider',
        ]
        self.processor = TrOCRProcessor.from_pretrained('raxtemur/trocr-base-ru')
        self.session_encoder = ort.InferenceSession(os.path.join(model_path, "encoder_model.onnx"), providers=providers)
        self.session_decoder = ort.InferenceSession(os.path.join(model_path, "decoder_model.onnx"), providers=providers)
        self.decoder_start_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.max_length = 64

    def predict(self, images):
        pix_val = self.processor(images=images, return_tensors='pt').pixel_values
        pix_val = pix_val.numpy()
        batch_size = pix_val.shape[0]
        encoded_list = self.session_encoder.run(['last_hidden_state'], {'pixel_values': pix_val})
        last_hidden_state = encoded_list[0]

        input_ids = np.array([[self.decoder_start_token_id] * batch_size], dtype=np.int64).reshape(batch_size, -1)
        eos_encountered = np.zeros(batch_size, dtype=np.bool)

        for _ in range(self.max_length):
            decoded_list = self.session_decoder.run(['logits'], {'input_ids': input_ids,
                                                                'encoder_hidden_states': last_hidden_state})
            gen_logits = decoded_list[0][:, -1, :]
            next_token_ids = np.argmax(gen_logits, axis=1)

            eos_encountered += next_token_ids == self.eos_token_id
            input_ids = np.concatenate([input_ids, next_token_ids.reshape(batch_size, -1)], axis=1)

            if np.all(eos_encountered):
                break

        return self.processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def predict_optimized(self, images):
        pix_val = self.processor(images=images, return_tensors='pt').pixel_values
        pix_val = pix_val.numpy()
        batch_size = pix_val.shape[0]
        encoded_list = self.session_encoder.run(['last_hidden_state'], {'pixel_values': pix_val})
        last_hidden_state = encoded_list[0]
        input_ids = np.array([[self.decoder_start_token_id] * batch_size], dtype=np.int64).reshape(batch_size, -1)
        eos_encountered = np.zeros(batch_size, dtype=np.bool)
        output_ids = np.zeros((batch_size, 0), dtype=np.int64)
        for _ in range(self.max_length):
            decoded_list = self.session_decoder.run(['logits'],
                                                    {'input_ids': input_ids,
                                                     'encoder_hidden_states': last_hidden_state[~eos_encountered, ...]})
            gen_logits = decoded_list[0][:, -1, :]
            next_token_ids = self.pad_token_id * np.ones(batch_size, dtype=np.int64)
            next_token_ids[~eos_encountered] = np.argmax(gen_logits, axis=1)

            eos_encountered += next_token_ids == self.eos_token_id
            output_ids = np.concatenate([output_ids, next_token_ids.reshape(batch_size, -1)], axis=1)
            input_ids = output_ids[~eos_encountered, :]

            if np.all(eos_encountered):
                break

        return self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)