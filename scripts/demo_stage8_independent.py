"""Demo using a REAL Stage-2 validation row; NOT a Teletalk customer."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.retention_recommender import RecommendationRequest, present
from src.stage8_integration import demo_independent_tracks

def main():
    p=argparse.ArgumentParser(description='Independent ML public benchmark + public Teletalk offer demo')
    p.add_argument('--track',required=True, choices=['iranian','ibm'])
    p.add_argument('--row-index',default=0,type=int)
    p.add_argument('--category',default='internet',choices=['internet','combo','minute','sms','unlimited_weekend'])
    p.add_argument('--budget',default=100,type=int)
    p.add_argument('--validity-days',default=7,type=int)
    a=p.parse_args()
    if a.row_index<0:p.error('row-index must be non-negative')
    from scripts.stage5_samples import validation_request
    sample=validation_request(a.track,row_index=a.row_index)  # real public dataset split
    offers=RecommendationRequest(category=a.category,budget_bdt=a.budget,
                                 exact_validity_days=a.validity_days)
    result=demo_independent_tracks(a.track,sample['features'],offers)
    pred=result.public_dataset_prediction
    print('PUBLIC DATASET DEMO:',pred['data_source_label'])
    print('Model:',pred['selected_model'],'| Churn score:',round(pred['churn_probability'],4),
          '| Threshold:',pred['decision_threshold'],'| Decision:',pred['predicted_label'])
    print('\nINDEPENDENT TELETALK PUBLIC OFFER MATCHES:')
    print(present(result.separate_public_offer_matches))
    print('\nIMPORTANT:',result.warning)
    return 0 if result.separate_public_offer_matches.status!='unavailable' else 1
if __name__=='__main__':sys.exit(main())
