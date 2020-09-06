import datetime
from sanic import response


def period_to_start_date(period):
    if 'D' in period:
        return datetime.datetime.today() - datetime.timedelta(days=int(period.split('D')[0]))
    elif 'M' in period:
        return datetime.datetime.today() - datetime.timedelta(days=int(period.split('M')[0])*31)
    elif 'Y' in period:
        return datetime.datetime.today() - datetime.timedelta(days=int(period.split('Y')[0])*365)
    else:
        raise Exception(f'Invalid period {period}')


def try_expt(msg, log=True, expt=Exception, err=Exception, pnt_original=False):
    def decorator(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except expt as e:
                if log:
                    args[0]._logger.error(msg)
                if pnt_original:
                    raise err(msg + f', reason: {str(e)}')
                else:
                    raise err(msg)
        return new_func
    return decorator


def web_expt():
    def decorator(func):
        async def new_func(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return response.json({'ret_code': 0, 'return': {'content': str(e)}})
        return new_func
    return decorator