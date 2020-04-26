from flask_restplus import Namespace, Resource, fields

from ..core.model import Model
from ..core.preprocessing import TextPreprocessing

api = Namespace('Business Objective Classification',
                description='Classification of US publicly listed companies based on business description')

api_model_payload = api.model('Payload',
                              {
                                  "input_text": fields.String(required=True,
                                                              description='Input Text (Business Description)')
                              })


@api.route('/api/bot/predict')
@api.expect(api_model_payload)
@api.doc("Post method which classifies the business "
         "objective to respective category")
class input_predict_text(Resource):

    def post(self):
        model = Model()
        input_string = self.api.payload['input_text']
        textprocessing_instance = TextPreprocessing(input_string)
        ip_seq = textprocessing_instance.tokenizer_and_pad()

        pred = model.pred_using_bilstm(ip_seq)
        return {'Result': pred}, 200
