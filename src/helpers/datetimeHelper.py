from datetime import datetime
import dateutil.relativedelta


class DateTimeHelper:

    @staticmethod
    def now():
        return datetime.now()

    @staticmethod
    def nowAsString():
        return datetime.now().strftime('%Y-%m-%d')

    @staticmethod
    def inPast(years=0, months=0, days=0):
        return datetime.now() - dateutil.relativedelta.relativedelta(years=years, months=months, days=days)

    @staticmethod
    def inPastAsString(years=0, months=0, days=0):
        return (datetime.now() - dateutil.relativedelta.relativedelta(years=years, months=months, days=days)).strftime(
            '%Y-%m-%d')

    @staticmethod
    def toString(date: datetime):
        return date.strftime('%Y-%m-%d')
