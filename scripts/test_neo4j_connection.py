import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables from .env
load_dotenv()

# Connection details
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))


def test_connection(driver):
    """
    Runs a simple query to verify the connection to Neo4j.
    """
    try:
        with driver.session() as session:
            # Run a simple Cypher query
            result = session.run("RETURN 'Connection successful!' AS message")

            # Extract the single record from the result.
            record = result.single()
            print(f" Test Query Result: {record['message']}")
            return True
    except Exception as e:
        print(f"Could not connect to the database or run query: {e}")
        return False


if __name__ == "__main__":
    print("Attempting to connect to Neo4j AuraDB...")

    if not all([URI, AUTH[0], AUTH[1]]):
        print("Critical Error: Missing database credential in .env file.")
        print(
            "   Please make sure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set."
        )
    else:
        # The Driver is the main entry point to the database.
        try:
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                print("Driver created successfully.")
                driver.verify_connectivity()
                print("Connection verified.")
                test_connection(driver)
        except Exception as e:
            print(f" Failed to create the driver or verufy connectivity : {e}")
