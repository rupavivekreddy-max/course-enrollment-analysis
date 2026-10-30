import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# ---------------------------------------
# Step 1 : Load Real Students
# ---------------------------------------
uploaded = files.upload()

if not uploaded:
    raise SystemExit("No CSV file selected.")

students_file = next(iter(uploaded))
students_df = pd.read_csv(students_file)

if "RollNo" not in students_df.columns or "StudentName" not in students_df.columns:
    raise SystemExit("CSV must contain 'RollNo' and 'StudentName' columns.")

students = list(zip(students_df["RollNo"], students_df["StudentName"]))

print("\nSelected Student CSV:", students_file)
print("Total Students Loaded:", len(students))


# NEW: read Branch and BatchYear straight from the Roll No.
# Example: 25B11AE001 -> BatchYear 2025, Branch "AE"
def roll_info(roll):
    roll = str(roll).strip().upper()
    batch_year = 2000 + int(roll[:2]) if roll[:2].isdigit() else np.nan
    branch = roll[5:7] if len(roll) >= 7 else "NA"
    return batch_year, branch


# Each course has TWO instructors plus a fixed duration in hours.
courses = {
    "Python Programming": {"instructors": ["Dr. K. Ramesh", "Ms. A. Kavitha"], "duration": 8},
    "Data Structures & Algorithms": {"instructors": ["Prof. S. Lakshmi", "Mr. B. Suresh"], "duration": 12},
    "Machine Learning Basics": {"instructors": ["Dr. A. Prasad", "Dr. N. Kalyani"], "duration": 10},
    "Web Development": {"instructors": ["Ms. R. Sindhu", "Mr. J. Varun"], "duration": 9},
    "Database Management Systems": {"instructors": ["Dr. V. Naidu", "Ms. C. Haritha"], "duration": 7},
    "Artificial Intelligence": {"instructors": ["Dr. M. Sarma", "Dr. P. Anitha"], "duration": 11},
    "Cloud Computing": {"instructors": ["Prof. T. Rao", "Mr. D. Vamsi"], "duration": 6},
    "Cyber Security Fundamentals": {"instructors": ["Ms. P. Devi", "Dr. S. Naresh"], "duration": 8},
}

months = ["Jan-2026", "Feb-2026", "Mar-2026", "Apr-2026", "May-2026", "Jun-2026"]

np.random.seed(42)

course_names = list(courses.keys())

instructor_quality = {}
for c, info in courses.items():
    for ins in info["instructors"]:
        instructor_quality[ins] = round(np.random.uniform(0.55, 0.95), 2)

CHEATER_PROBABILITY = 0.15
CERTIFICATE_WATCH_THRESHOLD = 90.0

rows = []

for roll, name in students:

    n_courses = np.random.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
    chosen = np.random.choice(course_names, size=n_courses, replace=False)

    batch_year, branch = roll_info(roll)  # NEW

    for course in chosen:

        info = courses[course]
        duration = info["duration"]
        instructor = np.random.choice(info["instructors"])
        quality = instructor_quality[instructor]
        month = np.random.choice(months)

        progress = int(np.clip(np.random.normal(quality * 100, 22), 0, 100))

        is_cheater = np.random.rand() < CHEATER_PROBABILITY and progress >= 80

        if is_cheater:
            progress = 100
            watched_fraction = np.random.uniform(0.05, 0.35)
        else:
            watched_fraction = np.clip(progress / 100 + np.random.normal(0, 0.05), 0, 1)

        watched_hours = round(watched_fraction * duration, 2)
        genuine_progress = round((watched_hours / duration) * 100, 1)

        if progress >= 100:
            status = "Completed"
            progress = 100
        elif progress >= 40:
            status = "In Progress"
        else:
            status = "Dropped"

        if status == "Completed":
            certificate = (
                "Issued"
                if genuine_progress >= CERTIFICATE_WATCH_THRESHOLD
                else "Withheld - Rewatch Required"
            )
        else:
            certificate = "Not Applicable"

        if genuine_progress >= 50:
            rating = round(float(np.clip(np.random.normal(quality * 5, 0.6), 1, 5)), 1)
        else:
            rating = np.nan

        rows.append({
            "RollNo": roll,
            "StudentName": name,
            "Branch": branch,                                    # NEW (13)
            "BatchYear": batch_year,                             # NEW (14)
            "Course": course,
            "Instructor": instructor,
            "EnrollmentMonth": month,
            "CourseDurationHours": duration,
            "WatchedHours": watched_hours,
            "Progress": progress,
            "GenuineProgress": genuine_progress,
            "ProgressGap": round(progress - genuine_progress, 1),  # NEW (15)
            "Status": status,
            "CertificateStatus": certificate,
            "StudentRating": rating,
        })

df = pd.DataFrame(rows)

# ---------------------------------------
# Step 2 : Save Dataset as CSV
# ---------------------------------------
df.to_csv("course_enrollment_data.csv", index=False)
print("course_enrollment_data.csv created successfully.\n")

# ---------------------------------------
# Step 3 : Preview Dataset
# ---------------------------------------
print("Online Course Enrollment Dataset (first 10 rows)\n")
print(df.head(10))

# ---------------------------------------
# Step 4 : Dataset Information
# ---------------------------------------
print("\n==============================")
print("Dataset Information")
print("==============================")
print("\nShape of Dataset :", df.shape)
print("\nColumn Names")
print(df.columns.tolist())
print("\nData Types")
print(df.dtypes)
print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------------
# Step 5 : Data Preprocessing
# ---------------------------------------
df.drop_duplicates(inplace=True)
df["Course"] = df["Course"].str.strip()
df["Status"] = df["Status"].str.strip()

# ---------------------------------------
# Step 6 : New Student Enrollment
# ---------------------------------------
print("\n==============================")
print("NEW COURSE ENROLLMENT")
print("==============================")

add_more = input("Do you want to enroll a new student in a course? (y/n): ").strip().lower()

while add_more == "y":

    roll = input("Enter Roll No : ").strip().upper()
    name = input("Enter Student Name : ").strip().upper()

    print("\nAvailable Courses :")
    for i, c in enumerate(course_names, start=1):
        print(f"{i}. {c} ({courses[c]['duration']} hrs)")

    course_choice = int(input("Choose Course Number : "))
    course_choice = max(1, min(len(course_names), course_choice))

    course = course_names[course_choice - 1]
    duration = courses[course]["duration"]
    instructors = courses[course]["instructors"]

    print(f"\nInstructors teaching '{course}' (based on past student outcomes):")

    course_df = df[df["Course"] == course]

    for idx, ins in enumerate(instructors, start=1):

        ins_df = course_df[course_df["Instructor"] == ins]
        n = len(ins_df)

        completion_rate = (
            round((ins_df["Status"] == "Completed").mean() * 100, 1) if n else 0.0
        )

        avg_rating = (
            round(ins_df["StudentRating"].mean(), 2)
            if ins_df["StudentRating"].notna().any()
            else None
        )

        rating_txt = f"{avg_rating}/5" if avg_rating is not None else "Not enough ratings yet"

        print(
            f"  {idx}. {ins:<18} "
            f"Completion Rate: {completion_rate:>5}%   "
            f"Senior Rating: {rating_txt}"
        )

    instructor_choice = int(input("Choose Instructor Number : "))
    instructor_choice = max(1, min(len(instructors), instructor_choice))
    instructor = instructors[instructor_choice - 1]

    print(f"\nAvailable Months : {', '.join(months)}")
    month = input("Enter Enrollment Month : ").strip()
    if month not in months:
        month = months[-1]

    progress_val = int(input("Enter Reported Progress % shown on the seek-bar (0-100) : "))
    progress_val = max(0, min(100, progress_val))

    watched_hours = float(input(f"Enter Genuine Watched Hours (0-{duration}) : "))
    watched_hours = max(0.0, min(float(duration), watched_hours))

    genuine_progress_val = round((watched_hours / duration) * 100, 1)

    if progress_val >= 100:
        status = "Completed"
    elif progress_val >= 40:
        status = "In Progress"
    else:
        status = "Dropped"

    if status == "Completed":
        certificate = (
            "Issued"
            if genuine_progress_val >= CERTIFICATE_WATCH_THRESHOLD
            else "Withheld - Rewatch Required"
        )
    else:
        certificate = "Not Applicable"

    batch_year, branch = roll_info(roll)  # NEW

    new_row = {
        "RollNo": roll,
        "StudentName": name,
        "Branch": branch,
        "BatchYear": batch_year,
        "Course": course,
        "Instructor": instructor,
        "EnrollmentMonth": month,
        "CourseDurationHours": duration,
        "WatchedHours": watched_hours,
        "Progress": progress_val,
        "GenuineProgress": genuine_progress_val,
        "ProgressGap": round(progress_val - genuine_progress_val, 1),
        "Status": status,
        "CertificateStatus": certificate,
        "StudentRating": np.nan,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    print(f"\n{name} ({roll}) successfully enrolled in '{course}' with status '{status}'.")

    if certificate == "Withheld - Rewatch Required":
        print(
            f"CERTIFICATE WITHHELD — only {genuine_progress_val}% genuinely watched. "
            f"Please rewatch the skipped content."
        )
    elif certificate == "Issued":
        print("Certificate ISSUED — genuine watch-time requirement met.")

    add_more = input("\nEnroll another student? (y/n): ").strip().lower()

df.to_csv("course_enrollment_data.csv", index=False)

print("\nUpdated dataset saved to course_enrollment_data.csv")
print("Current Dataset Shape :", df.shape)

# ---------------------------------------
# Step 7 : Course Popularity Analysis
# ---------------------------------------
print("\n==============================")
print("COURSE POPULARITY (Enrollments per Course)")
print("==============================")

course_popularity = df["Course"].value_counts()
print(course_popularity)

# ---------------------------------------
# Step 8 : Completion Rate Calculation
# ---------------------------------------
print("\n==============================")
print("COMPLETION STATUS BREAKDOWN")
print("==============================")

status_counts = df["Status"].value_counts()
print(status_counts)

overall_completion_rate = (df["Status"] == "Completed").mean() * 100
print(f"\nOverall Completion Rate : {overall_completion_rate:.2f}%")

print("\nCompletion Rate by Course")
completion_by_course = df.groupby("Course")["Status"].apply(
    lambda s: round((s == "Completed").mean() * 100, 2)
)
print(completion_by_course)

# ---------------------------------------
# Step 9 : Instructor-wise Enrollment
# ---------------------------------------
print("\n==============================")
print("INSTRUCTOR-WISE ENROLLMENT")
print("==============================")

instructor_enrollment = df["Instructor"].value_counts()
print(instructor_enrollment)

# ---------------------------------------
# Step 10 : Instructor Recommendation
# ---------------------------------------
print("\n==============================")
print("INSTRUCTOR RECOMMENDATION REPORT (for new/junior students)")
print("==============================")

recommendation_rows = []

for course, info in courses.items():

    course_df = df[df["Course"] == course]

    for ins in info["instructors"]:

        ins_df = course_df[course_df["Instructor"] == ins]
        n = len(ins_df)

        completion_rate = (
            round((ins_df["Status"] == "Completed").mean() * 100, 1) if n else 0.0
        )

        avg_rating = (
            round(ins_df["StudentRating"].mean(), 2)
            if ins_df["StudentRating"].notna().any()
            else np.nan
        )

        recommendation_rows.append({
            "Course": course,
            "Instructor": ins,
            "Students": n,
            "CompletionRate%": completion_rate,
            "AvgSeniorRating": avg_rating,
        })

recommendation_df = pd.DataFrame(recommendation_rows).sort_values(
    ["Course", "CompletionRate%"], ascending=[True, False]
)

print(recommendation_df.to_string(index=False))

# ---------------------------------------
# Step 11 : Certificate Integrity Report
# ---------------------------------------
print("\n==============================")
print("CERTIFICATE INTEGRITY REPORT (Anti-Cheat)")
print("==============================")

cert_counts = df["CertificateStatus"].value_counts()
print(cert_counts)

withheld = df[df["CertificateStatus"] == "Withheld - Rewatch Required"]

if len(withheld):
    print(
        f"\n{len(withheld)} certificate(s) withheld due to insufficient "
        f"genuine watch-time:"
    )
    print(
        withheld[
            ["RollNo", "StudentName", "Course", "Progress", "GenuineProgress", "ProgressGap"]
        ].to_string(index=False)
    )
else:
    print("\nNo certificates withheld — all completions had genuine watch-time.")

# ProgressGap column now holds this per row
gap = df["ProgressGap"].mean()
print(
    f"\nAverage gap between reported progress and genuine watch-time "
    f"progress: {gap:.2f} percentage points"
)

# ---------------------------------------
# Step 12 : Student Progress Analysis
# ---------------------------------------
print("\n==============================")
print("NUMPY - STUDENT PROGRESS ANALYSIS")
print("==============================")

progress = df["Progress"].values

print("Average Progress :", np.mean(progress))
print("Maximum Progress :", np.max(progress))
print("Minimum Progress :", np.min(progress))
print("Median Progress :", np.median(progress))
print("Standard Deviation :", np.std(progress))

student_avg_progress = (
    df.groupby("StudentName")["Progress"].mean().sort_values(ascending=False)
)

print("\nTop 5 Students by Average Progress")
print(student_avg_progress.head(5))

print("\nBottom 5 Students by Average Progress")
print(student_avg_progress.tail(5))

# ---------------------------------------
# Step 13 : Pandas Statistical Summary
# ---------------------------------------
print("\n==============================")
print("PANDAS ANALYSIS")
print("==============================")

print(df.select_dtypes(include=[np.number]).describe())

# ---------------------------------------
# Step 14 : Bar Chart
# ---------------------------------------
plt.figure(figsize=(8, 5))
course_popularity.plot(kind="bar", color="steelblue")
plt.title("Course-wise Enrollment Count")
plt.xlabel("Course")
plt.ylabel("Number of Enrollments")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("chart_course_enrollments.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 15 : Pie Chart
# ---------------------------------------
# FIX: colours are now matched to the status name, so
# "Completed" is always green even if value_counts() order changes.
status_color_map = {
    "Completed": "#4CAF50",
    "In Progress": "#FFC107",
    "Dropped": "#F44336",
}

plt.figure(figsize=(6, 6))
status_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=[status_color_map.get(s, "#9E9E9E") for s in status_counts.index],
)
plt.title("Completion Status Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart_completion_status.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 16 : Monthly Enrollment Trend
# ---------------------------------------
monthly_enrollments = df["EnrollmentMonth"].value_counts().reindex(months)

plt.figure(figsize=(8, 5))
plt.plot(
    monthly_enrollments.index,
    monthly_enrollments.values,
    marker="o",
    color="darkorange",
)
plt.title("Monthly Enrollment Trend")
plt.xlabel("Month")
plt.ylabel("Number of Enrollments")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("chart_monthly_enrollments.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 17 : Instructor-wise Bar Chart
# ---------------------------------------
plt.figure(figsize=(9, 5))
instructor_enrollment.plot(kind="bar", color="mediumpurple")
plt.title("Instructor-wise Enrollment Count")
plt.xlabel("Instructor")
plt.ylabel("Number of Students")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("chart_instructor_enrollments.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 18 : Certificate Status Chart
# ---------------------------------------
plt.figure(figsize=(6, 6))

cert_color_map = {
    "Issued": "#4CAF50",
    "Withheld - Rewatch Required": "#F44336",
    "Not Applicable": "#9E9E9E",
}

cert_colors = [cert_color_map.get(lbl, "#9E9E9E") for lbl in cert_counts.index]

cert_counts.plot(kind="pie", autopct="%1.1f%%", colors=cert_colors)
plt.title("Certificate Status (Anti-Cheat Check)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("chart_certificate_status.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 19 : Instructor Rating Chart
# ---------------------------------------
rating_by_instructor = (
    df.groupby("Instructor")["StudentRating"].mean().dropna().sort_values(ascending=False)
)

plt.figure(figsize=(9, 5))
rating_by_instructor.plot(kind="bar", color="teal")
plt.title("Average Senior Rating by Instructor")
plt.xlabel("Instructor")
plt.ylabel("Average Rating (out of 5)")
plt.ylim(0, 5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("chart_instructor_ratings.png", dpi=150)
plt.show()

# ---------------------------------------
# Step 20 : Completion Prediction
# ---------------------------------------
print("\n==============================")
print("COURSE COMPLETION PREDICTION (Enhancement)")
print("==============================")

progress_in = int(input("Enter Student Progress % : "))
days_left = int(input("Enter Days Left in Course : "))

if progress_in >= 80 or (progress_in >= 50 and days_left >= 15):
    print("\nPrediction : LIKELY TO COMPLETE")
elif progress_in >= 40:
    print("\nPrediction : AT RISK")
else:
    print("\nPrediction : LIKELY TO DROP")

print("\nProject Completed Successfully.")
