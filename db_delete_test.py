import psycopg2

DB_CONFIG = {
    "user": "ai4xadmin",
    "password": "password123#",  # replace with env var in production
    "host": "ai4x.postgres.database.azure.com",
    "port": 5432,
    "database": "ai4xdemo"
}

TABLE_NAME = "ai4x_demo_requests_log_v4"

def delete_customers():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = f"""
        DELETE FROM {TABLE_NAME}
        WHERE customerName IN (%s, %s)
    """
    cur.execute(query, ("cust1", "cust2"))

    conn.commit()
    cur.close()
    conn.close()
    print("Deleted entries for cust1 and cust2")

def delete_all_entries():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = f"DELETE FROM {TABLE_NAME}"
    cur.execute(query)

    conn.commit()
    cur.close()
    conn.close()
    print(f"All entries deleted from {TABLE_NAME}")

def delete_by_customer_app():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query = f"""
        DELETE FROM {TABLE_NAME}
        WHERE customerApp = %s
    """
    cur.execute(query, ("FleetAnalyticsApp",))

    conn.commit()
    cur.close()
    conn.close()
    print("Deleted entries where customerApp = 'FleetAnalyticsApp'")

if __name__ == "__main__":
    delete_by_customer_app()
