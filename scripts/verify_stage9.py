"""Stage 9 offline integration verification, backed by existing public validation samples."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import pytest
from src.dashboard_stage9 import (
    checked_prediction,checked_explanation,local_report,public_validation_sample,
    input_form_fields,public_validation_count,model_scorecard,
)

def main():
    print('Stage 9 verified-model dashboard integration (NO website or LLM calls).')
    for track in ('iranian','ibm'):
        row=public_validation_sample(track,0)
        fields=input_form_fields(track,row)
        n=public_validation_count(track)
        pred=checked_prediction(track,row)
        if pred['track']!=track:raise AssertionError('Track mismatch')
        shap=checked_explanation(track,row)
        if abs(shap['probability_churn']-pred['churn_probability'])>5e-5:
            raise AssertionError('Model and SHAP probabilities mismatch')
        card=model_scorecard(track)
        assert card['track']==track
        assert len(fields)==len(local_report(track)['input_features'])
        assert n>0
        print(f"PASS: {track} | validation rows={n} | strict fields={len(fields)} | score={pred['churn_probability']:.5f} | SHAP consistent")
    print('SUCCESS: Stage 9 saved models, validation input and local SHAP integration verified.')

if __name__=='__main__':main()
