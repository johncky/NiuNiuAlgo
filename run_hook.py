if __name__ == '__main__':
    import os
    import sys
    cwd = os.getcwd() + '/FutuAlgo'
    if cwd not in sys.path:
        sys.path.append(cwd)

    import FutuAlgo

    # edit your own
    INIT_DATATYPE = ['K_DAY']
    INIT_TICKERS = ['HK.00700']

    futu_hook = FutuAlgo.FutuHook()
    futu_hook.subscribe(datatypes=INIT_DATATYPE, tickers=INIT_TICKERS)
    futu_hook.run()
