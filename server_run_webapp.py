if __name__ == '__main__':
    import os
    import sys
    cwd = os.getcwd() + '/FutuAlgo'
    if cwd not in sys.path:
        sys.path.append(cwd)

    if sys.platform == 'linux2':
        import ctypes

        libc = ctypes.cdll.LoadLibrary('libc.so.6')
        libc.prctl('PR_SET_NAME ', 'FutuHook', 0, 0, 0)

    import FutuAlgo

    app = FutuAlgo.WebApp()
    # name & sanic ip address of the running algo
    app.run(8522, hook_ip='http://0.0.0.0:8000')
