"""Customer-facing live-official Teletalk Q&A and Stage 5/6 unified demo."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.teletalk_full_copilot import ask_project, CHURN_WORDS


def main(argv=None):
    p=argparse.ArgumentParser(description='Stage 7 LIVE official Teletalk info (no paid search API)')
    p.add_argument('--question',default='100 takar moddhe 7 diner internet pack')
    p.add_argument('--chat', action='store_true')
    p.add_argument('--track',choices=('iranian','ibm'))
    p.add_argument('--sample-index',type=int,default=0)
    p.add_argument('--live',action='store_true',help='Enable Stage 6 live Gemini/Groq for CHURN; offers always use website')
    args=p.parse_args(argv)
    if args.sample_index<0:p.error('sample-index must be >= 0')

    def answer(q):
        is_churn=any(word in q.casefold() for word in CHURN_WORDS)
        features=None
        if is_churn and args.track:
            from scripts.stage5_samples import validation_request
            features=validation_request(args.track,row_index=args.sample_index)['features']
        result=ask_project(q,track=args.track,features=features,live=args.live)
        print('\n'+result.message)
        print(f'\nMODE: {result.mode} | PROVIDER: {getattr(result,"provider","verified-local")}')
        for url in getattr(result,'citations',()):print('OFFICIAL SOURCE:',url)
        return result
    if args.chat:
        print('TeleRetain AI: live public Teletalk pages + separate public-dataset churn demo. Type exit to quit.')
        while True:
            try:q=input('You> ').strip()
            except (KeyboardInterrupt,EOFError):break
            if q.lower() in ('quit','exit'):break
            if q:answer(q)
    else:
        result=answer(args.question)
        if result.mode=='unavailable':return 1
    return 0

if __name__=='__main__':sys.exit(main())
