import pyodbc
import argparse
import sys


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username')
    parser.add_argument('-p', '--password')
    args = parser.parse_args()

    server = 'RESWNCKSQPR02.research.chop.edu'
    database = 'Reporting_eSPA'

    if sys.platform == "win32":
        cnxn = pyodbc.connect(
            'DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' +
            args.username + ';PWD=' + args.password)
        # cnxn = pyodbc.connect(
        #     'DRIVER={ODBC Driver 13 for SQL Server};SERVER=' + server + ';DATABASE=' + database +
        #     ';Trusted_Connection=yes')

    else:
        # cnxn = pyodbc.connect('DSN=MYMSSQL;UID=' + args.username + ';PWD=' + args.password)
        cnxn = pyodbc.connect('Trusted_Connection=yes; DSN=MYMSSQL')

    cursor = cnxn.cursor()
    rows = cursor.execute('''SELECT top 10 OID, Funding_Proposal_ID, Name, Opportunity_ID, CFDA, Competition_ID, 
                        Open_Date, Close_Date, Is_Supported, Agency 
                        FROM Reporting_eSPA.dbo.Funding_Opportunities''').fetchall()

    print(rows)


if __name__ == '__main__':
    main()

