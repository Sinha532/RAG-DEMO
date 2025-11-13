"""
Database module for patient management system
Creates and manages SQLite database with patient, medical records, and appointments
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random


def create_patient_database():
    """Create and populate the patient database with dummy data"""
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth DATE NOT NULL,
            gender TEXT NOT NULL,
            blood_group TEXT NOT NULL,
            contact_no TEXT NOT NULL,
            email TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            emergency_contact_name TEXT,
            emergency_contact_no TEXT,
            created_date DATE DEFAULT CURRENT_DATE
        )
    ''')
    
    # Create medical_records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            diagnosis TEXT NOT NULL,
            symptoms TEXT,
            prescribed_medication TEXT,
            doctor_name TEXT NOT NULL,
            visit_date DATE NOT NULL,
            next_appointment DATE,
            notes TEXT,
            is_diabetic BOOLEAN DEFAULT 0,
            has_bp BOOLEAN DEFAULT 0,
            allergies TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')
    
    # Create appointments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            doctor_name TEXT NOT NULL,
            appointment_date DATE NOT NULL,
            appointment_time TEXT NOT NULL,
            department TEXT NOT NULL,
            status TEXT DEFAULT 'Scheduled',
            reason TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) FROM patients')
    if cursor.fetchone()[0] > 0:
        conn.close()
        return "Database already exists with data"
    
    # Insert dummy patient data
    patients_data = [
        ('Rajesh', 'Kumar', '1985-03-15', 'Male', 'O+', '9876543210', 'rajesh.kumar@email.com',
         '123 MG Road', 'Bangalore', 'Karnataka', 'Priya Kumar', '9876543211'),
        ('Priya', 'Sharma', '1990-07-22', 'Female', 'A+', '9876543212', 'priya.sharma@email.com',
         '456 Park Street', 'Mumbai', 'Maharashtra', 'Amit Sharma', '9876543213'),
        ('Amit', 'Patel', '1978-11-08', 'Male', 'B+', '9876543214', 'amit.patel@email.com',
         '789 Gandhi Nagar', 'Ahmedabad', 'Gujarat', 'Neha Patel', '9876543215'),
        ('Neha', 'Singh', '1995-05-30', 'Female', 'AB+', '9876543216', 'neha.singh@email.com',
         '321 Lake View', 'Delhi', 'Delhi', 'Vikram Singh', '9876543217'),
        ('Vikram', 'Reddy', '1982-09-12', 'Male', 'O-', '9876543218', 'vikram.reddy@email.com',
         '654 Beach Road', 'Chennai', 'Tamil Nadu', 'Lakshmi Reddy', '9876543219'),
        ('Lakshmi', 'Iyer', '1988-01-25', 'Female', 'A-', '9876543220', 'lakshmi.iyer@email.com',
         '987 Temple Street', 'Hyderabad', 'Telangana', 'Ramesh Iyer', '9876543221'),
        ('Ramesh', 'Gupta', '1975-12-03', 'Male', 'B-', '9876543222', 'ramesh.gupta@email.com',
         '147 River View', 'Pune', 'Maharashtra', 'Anjali Gupta', '9876543223'),
        ('Anjali', 'Mehta', '1992-08-17', 'Female', 'AB-', '9876543224', 'anjali.mehta@email.com',
         '258 Hill Station', 'Kolkata', 'West Bengal', 'Rohan Mehta', '9876543225'),
        ('Rohan', 'Nair', '1980-04-28', 'Male', 'O+', '9876543226', 'rohan.nair@email.com',
         '369 Garden Road', 'Kochi', 'Kerala', 'Divya Nair', '9876543227'),
        ('Divya', 'Chopra', '1993-10-14', 'Female', 'A+', '9876543228', 'divya.chopra@email.com',
         '741 Market Street', 'Jaipur', 'Rajasthan', 'Karan Chopra', '9876543229'),
        ('Sandeep', 'Jain', '1987-02-18', 'Male', 'B+', '9876543230', 'sandeep.jain@email.com',
         '159 Crescent Road', 'Bangalore', 'Karnataka', 'Pooja Jain', '9876543231'),
        ('Pooja', 'Mishra', '1991-11-20', 'Female', 'O-', '9876543232', 'pooja.mishra@email.com',
         '753 Lotus Lane', 'Pune', 'Maharashtra', 'Sandeep Jain', '9876543233'),
        ('Karan', 'Malhotra', '1979-06-05', 'Male', 'A-', '9876543234', 'karan.malhotra@email.com',
         '852 Sunrise Apartments', 'Delhi', 'Delhi', 'Deepa Malhotra', '9876543235'),
        ('Deepa', 'Verma', '1983-09-25', 'Female', 'AB+', '9876543236', 'deepa.verma@email.com',
         '963 Palm Grove', 'Mumbai', 'Maharashtra', 'Karan Malhotra', '9876543237'),
        ('Arjun', 'Rao', '1996-01-10', 'Male', 'O+', '9876543238', 'arjun.rao@email.com',
         '147 Sterling Towers', 'Hyderabad', 'Telangana', 'Meera Rao', '9876543239'),
        ('Meera', 'Krishnan', '1994-04-03', 'Female', 'B-', '9876543240', 'meera.krishnan@email.com',
         '258 Pearl Residency', 'Chennai', 'Tamil Nadu', 'Arjun Rao', '9876543241'),
        ('Suresh', 'Bose', '1972-07-19', 'Male', 'A+', '9876543242', 'suresh.bose@email.com',
         '369 Victoria Garden', 'Kolkata', 'West Bengal', 'Mala Bose', '9876543243'),
        ('Mala', 'Sen', '1976-12-28', 'Female', 'O+', '9876543244', 'mala.sen@email.com',
         '741 Howrah Bridge', 'Kolkata', 'West Bengal', 'Suresh Bose', '9876543245'),
        ('Vivek', 'Anand', '1989-08-08', 'Male', 'AB-', '9876543246', 'vivek.anand@email.com',
         '852 Silicon Oasis', 'Bangalore', 'Karnataka', 'Sunita Anand', '9876543247'),
        ('Sunita', 'Narayanan', '1990-03-12', 'Female', 'B+', '9876543248', 'sunita.narayanan@email.com',
         '963 Electronic City', 'Bangalore', 'Karnataka', 'Vivek Anand', '9876543249')
    ]
    
    cursor.executemany('''
        INSERT INTO patients (first_name, last_name, date_of_birth, gender, blood_group,
                            contact_no, email, address, city, state,
                            emergency_contact_name, emergency_contact_no)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', patients_data)
    
    # Insert dummy medical records
    diagnoses = ['Type 2 Diabetes', 'Hypertension', 'Common Cold', 'Migraine', 'Asthma',
                 'Arthritis', 'Gastritis', 'Anxiety', 'Back Pain', 'Allergic Rhinitis', 'PCOS', 'GERD']
    medications = ['Metformin', 'Amlodipine', 'Paracetamol', 'Ibuprofen', 'Salbutamol',
                   'Aspirin', 'Omeprazole', 'Sertraline', 'Diclofenac', 'Cetirizine', 'Myo-Inositol', 'Pantoprazole']
    doctors = ['Dr. Ramesh Kumar', 'Dr. Priya Sharma', 'Dr. Amit Patel', 'Dr. Sneha Reddy', 
               'Dr. Vikram Singh', 'Dr. Alok Gupta', 'Dr. Meena Iyer']
    
    medical_records = []
    for patient_id in range(1, 21):
        for _ in range(random.randint(2, 5)):
            diagnosis = random.choice(diagnoses)
            medication = random.choice(medications)
            doctor = random.choice(doctors)
            visit_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            next_appointment = (datetime.now() + timedelta(days=random.randint(30, 120))).strftime('%Y-%m-%d')
            is_diabetic = 1 if 'Diabetes' in diagnosis else random.choice([0, 0, 0, 1])
            has_bp = 1 if 'Hypertension' in diagnosis else random.choice([0, 0, 0, 1])
            
            medical_records.append((
                patient_id, diagnosis, f'Symptoms related to {diagnosis}', medication,
                doctor, visit_date, next_appointment, f'Follow-up recommended for {diagnosis}',
                is_diabetic, has_bp, 'None' if random.random() > 0.3 else 'Penicillin'
            ))
    
    cursor.executemany('''
        INSERT INTO medical_records (patient_id, diagnosis, symptoms, prescribed_medication,
                                    doctor_name, visit_date, next_appointment, notes,
                                    is_diabetic, has_bp, allergies)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', medical_records)
    
    # Insert dummy appointments
    departments = ['Cardiology', 'General Medicine', 'Orthopedics', 'Neurology', 
                   'Pediatrics', 'Gynecology', 'Endocrinology']
    times = ['09:00 AM', '10:30 AM', '02:00 PM', '03:30 PM', '05:00 PM']
    statuses = ['Scheduled', 'Completed', 'Cancelled', 'No Show']
    
    appointments = []
    for patient_id in range(1, 21):
        for _ in range(random.randint(1, 4)):
            appointment_date = (datetime.now() + timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
            appointments.append((
                patient_id, random.choice(doctors), appointment_date, random.choice(times),
                random.choice(departments), random.choice(statuses), 'Regular checkup'
            ))
    
    cursor.executemany('''
        INSERT INTO appointments (patient_id, doctor_name, appointment_date, appointment_time,
                                department, status, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', appointments)
    
    conn.commit()
    conn.close()
    
    return "✅ Database created successfully with 20 patients and medical records!"


def query_database(sql_query):
    """Execute a SQL query and return the result as a DataFrame"""
    try:
        conn = sqlite3.connect('patients.db')
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except sqlite3.OperationalError as e:
        return f"Database query error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


def get_database_schema():
    """Get the schema of all tables in the database"""
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    
    schema_info = ""
    
    # Get schema for patients table
    cursor.execute("PRAGMA table_info(patients)")
    patients_schema = cursor.fetchall()
    schema_info += "TABLE: patients\nCOLUMNS:\n"
    for col in patients_schema:
        schema_info += f"  - {col[1]} ({col[2]})\n"
    schema_info += "\n"
    
    # Get schema for medical_records table
    cursor.execute("PRAGMA table_info(medical_records)")
    medical_records_schema = cursor.fetchall()
    schema_info += "TABLE: medical_records\nCOLUMNS:\n"
    for col in medical_records_schema:
        schema_info += f"  - {col[1]} ({col[2]})\n"
    schema_info += "\n"
    
    # Get schema for appointments table
    cursor.execute("PRAGMA table_info(appointments)")
    appointments_schema = cursor.fetchall()
    schema_info += "TABLE: appointments\nCOLUMNS:\n"
    for col in appointments_schema:
        schema_info += f"  - {col[1]} ({col[2]})\n"
    
    conn.close()
    return schema_info


def get_patient_info():
    """Get sample patient info from the database"""
    try:
        conn = sqlite3.connect('patients.db')
        df = pd.read_sql_query(
            "SELECT patient_id, first_name, last_name, city, state, contact_no FROM patients LIMIT 10", 
            conn
        )
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame(columns=["Error"], data=[[f"Could not retrieve patient info: {e}"]])


if __name__ == "__main__":
    result = create_patient_database()
    print(result)
