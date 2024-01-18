import pyodbc

# Connection parameters
server = 'RAVI-DESKTOP\SQLEXPRESS01'
database = 'SentiRec_Analytics'
username = 'RAVI-DESKTOP\RaviB'
driver = '{SQL Server}'

# Create a connection string
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

try:
    # Establish a connection
    connection = pyodbc.connect(connection_string)

    # Create a cursor from the connection
    cursor = connection.cursor()

    # Execute a simple query
    cursor.execute("SELECT 1")

    # Fetch the result
    result = cursor.fetchone()

    # Print the result
    print("Connection successful. Result:", result)

except pyodbc.Error as e:
    print("Error connecting to the database:", e)

finally:
    # Close the cursor and connection
    cursor.close()
    connection.close()
