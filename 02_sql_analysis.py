import os
import sqlite3
import pandas as pd


DATA_PATH = "data/processed/credit_clean.csv"
DB_PATH = "database/credit_risk.db"
OUTPUT_DIR = "reports/sql_outputs"


def run_query(conn, query, output_name):
    result = pd.read_sql_query(query, conn)

    output_path = os.path.join(OUTPUT_DIR, output_name)
    result.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(result.head())
    print("-" * 80)


def main():
    os.makedirs("database", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Cleaned data not found. Run python 01_clean_data.py first."
        )

    df = pd.read_csv(DATA_PATH)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("credit_customers", conn, if_exists="replace", index=False)

    print("Data loaded into SQLite database:", DB_PATH)

    queries = {
        "default_by_age_group.csv": """
            SELECT 
                age_group,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate
            FROM credit_customers
            GROUP BY age_group
            ORDER BY default_rate DESC;
        """,

        "default_by_credit_limit.csv": """
            SELECT 
                limit_bucket,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate,
                ROUND(AVG(limit_bal), 2) AS avg_credit_limit
            FROM credit_customers
            GROUP BY limit_bucket
            ORDER BY default_rate DESC;
        """,

        "default_by_education.csv": """
            SELECT 
                education,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate
            FROM credit_customers
            GROUP BY education
            ORDER BY default_rate DESC;
        """,

        "default_by_marriage.csv": """
            SELECT 
                marriage,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate
            FROM credit_customers
            GROUP BY marriage
            ORDER BY default_rate DESC;
        """,

        "default_by_delay_months.csv": """
            SELECT 
                months_with_delay,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate
            FROM credit_customers
            GROUP BY months_with_delay
            ORDER BY months_with_delay;
        """,

        "high_risk_segments.csv": """
            SELECT 
                age_group,
                limit_bucket,
                education,
                COUNT(*) AS total_customers,
                SUM(is_default) AS default_customers,
                ROUND(100.0 * SUM(is_default) / COUNT(*), 2) AS default_rate
            FROM credit_customers
            GROUP BY age_group, limit_bucket, education
            HAVING total_customers >= 100
            ORDER BY default_rate DESC
            LIMIT 15;
        """
    }

    for output_name, query in queries.items():
        run_query(conn, query, output_name)

    conn.close()


if __name__ == "__main__":
    main()