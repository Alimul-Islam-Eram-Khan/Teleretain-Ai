"""Interactive unified Stage 6/7/8 research-only chat (CLI; not the future Streamlit UI)."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.stage8_chat import ask_research_copilot

def main():
    parser=argparse.ArgumentParser(description='Teletalk public-source offers + separate public-data churn demo')
    parser.add_argument('--track',choices=['iranian','ibm'],help='Only needed for public-dataset churn questions')
    parser.add_argument('--sample-index',type=int,default=0)
    parser.add_argument('--live-llm',action='store_true',help='Optional Gemini/Groq only for public-dataset churn explanation')
    parser.add_argument('--question',default=None,help='Omit for interactive chat')
    parser.add_argument('--language',choices=['english','bangla','banglish'],default='banglish')
    args=parser.parse_args()
    if args.sample_index<0:parser.error('sample-index must be nonnegative')

    def reply(question):
        from src.teletalk_full_copilot import CHURN_WORDS
        features=None
        if args.track and any(word in question.casefold() for word in CHURN_WORDS):
            from scripts.stage5_samples import validation_request
            features=validation_request(args.track,row_index=args.sample_index)['features']
        status,content,cites=ask_research_copilot(question,track=args.track,features=features,
            live_llm=args.live_llm,language=args.language)
        print('\n'+content+'\nMODE:',status)
        for source in cites:print('OFFICIAL SOURCE:',source)
        return status

    print('RESEARCH POC: public Teletalk offers; independent Iranian/IBM benchmark churn. No private customer data.')
    if args.question:
        status=reply(args.question)
        return 1 if status=='unavailable' else 0
    print('Type exit to finish; do not enter private NID, phone numbers or customer billing data.')
    while True:
        try:question=input('You> ').strip()
        except (KeyboardInterrupt,EOFError):break
        if question.lower() in ('exit','quit'):break
        if question: reply(question)
    return 0
if __name__=='__main__':sys.exit(main())
