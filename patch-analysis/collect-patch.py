import subprocess
import re
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

# Directory paths
REPO_PATH = "../rust-linux/linux"  # Path to the Linux kernel repository
OUTPUT_DIR = "patch_analysis"  # Directory to store analysis outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simplified keywords for Git --grep
KEYWORDS = [
    "memory leak", "mem leak", "leak memory",
    "use-after-free", "use after free", "UAF",
    "buffer overflow", "buf overflow", "overrun buffer",
    "dangling pointer", "dangling ptr",
    "race condition", "race cond", "data race",
    "CVE", "vulnerability", "exploit",
    "privilege escalation",
    "fixes:", "addresses:", "patch",
    "null pointer", "nullptr", "nullptr deref", "NULL deref",
    "safe", "unsafe"
]

# Regex patterns for categorization
CATEGORIES = {
    "Memory Leak": re.compile(r"memory[ _-]?leak", re.IGNORECASE),
    "Use-After-Free": re.compile(r"use[ _-]?after[ _-]?free", re.IGNORECASE),
    "Buffer Overflow": re.compile(r"buffer[ _-]?overflow", re.IGNORECASE),
    "Dangling Pointer": re.compile(r"dangling[ _-]?pointer", re.IGNORECASE),
    "Race Condition": re.compile(r"race[ _-]?condition", re.IGNORECASE),
    "CVE": re.compile(r"CVE[ _-]?\d{4}-\d+", re.IGNORECASE),
    "Vulnerability": re.compile(r"vulnerability", re.IGNORECASE),
    "Exploit": re.compile(r"exploit", re.IGNORECASE),
    "Privilege Escalation": re.compile(r"privilege[ _-]?escalation", re.IGNORECASE),
    "Null Pointer": re.compile(r"null[ _-]?pointer|nullptr", re.IGNORECASE),
    "Fixes": re.compile(r"fixes:", re.IGNORECASE),
    "Addresses": re.compile(r"addresses:", re.IGNORECASE),
    "Patch": re.compile(r"patch", re.IGNORECASE),
}

# Rust language features for bug prevention
RUST_FEATURES = {
    "Memory Leak": "Ownership, Borrow Checker",
    "Use-After-Free": "Lifetimes, Ownership",
    "Buffer Overflow": "Boundary Checks",
    "Dangling Pointer": "Ownership, Borrow Checker",
    "Race Condition": "Send/Sync Traits, Borrow Checker",
    "Null Pointer": "Null Safety",
}

def run_git_command(command_args, repo_path):
    """Run a Git command and return its output."""
    print(f"Running command: git {' '.join(command_args)}")
    result = subprocess.run(
        ["git"] + command_args,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"Git command failed: {result.stderr}")
        return ""
    print(f"Command output (truncated): {result.stdout[:500]}...")
    return result.stdout

def extract_commits_by_keywords(repo_path, keywords, since="2022-8-01", until="2024-11-01"):
    """Extract commits matching specific keywords within a given time frame."""
    grep_args = []
    for keyword in keywords:
        grep_args.extend(["--grep", keyword])

    log_command = ["log", "--regexp-ignore-case", "--format=%H %ad %s", "--date=short"] + grep_args + ["--since", since, "--until", until, "HEAD", "--"]
    log_output = run_git_command(log_command, repo_path)

    with open(f"{OUTPUT_DIR}/filtered_commits.log", "w") as f:
        f.write(log_output)

    num_commits = len(log_output.splitlines())
    print(f"Found {num_commits} commits matching the keywords.")
    return log_output

def categorize_commits(commits_log):
    """Categorize commits using regex patterns."""
    results = {category: 0 for category in CATEGORIES}
    matches = {category: [] for category in CATEGORIES}

    for line in commits_log.splitlines():
        for category, pattern in CATEGORIES.items():
            if pattern.search(line):
                results[category] += 1
                matches[category].append(line)
                break

    for category, count in results.items():
        print(f"Category '{category}': {count} commits.")
    return results, matches

def adjust_time_axis(df, ax):
    """Adjust x-axis to a fixed time range from November 2022 to November 2024."""
    # Define fixed date range
    min_date = datetime(2022, 8, 1)
    max_date = datetime(2024, 8, 1)  # End of November 2024
    ax.set_xlim(min_date, max_date)
    
    # Use MonthLocator to show months, format with abbreviated names
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")

def plot_bug_trends_over_time(filtered_commits_log):
    """Plot bug trends over time for selected bug categories."""
    selected_categories = [
        "Memory Leak", "Dangling Pointer", "Null Pointer", 
        "Use-After-Free", "Buffer Overflow"
    ]
    dates = []
    categories = []

    with open(filtered_commits_log, "r") as f:
        for line in f:
            for category, pattern in CATEGORIES.items():
                if pattern.search(line):
                    try:
                        date = line.split()[1]
                        dates.append(datetime.strptime(date, "%Y-%m-%d"))
                        categories.append(category)
                    except (ValueError, IndexError):
                        continue

    df = pd.DataFrame({"date": dates, "category": categories})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        print("No data available for bug trends over time.")
        return

    # Filter for selected categories only
    df = df[df["category"].isin(selected_categories)]

    # Group by month and category
    df = df.groupby([pd.Grouper(key="date", freq="MS"), "category"]).size().unstack(fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    df.plot(kind="line", ax=ax, marker="o")
    adjust_time_axis(df, ax)
    plt.title("Bug Trends Over Time", fontsize=16)
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Number of Bugs", fontsize=12)
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Bug Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bug_trends_over_time.png")
    plt.close()


def plot_stacked_distribution(filtered_commits_log):
    """Plot the stacked distribution of bug categories over time as percentages."""
    # Reordered stack
    selected_categories = [
        "Dangling Pointer", "Buffer Overflow", 
        "Use-After-Free", "Null Pointer", "Memory Leak"
    ]

    # Softer color palette
    category_colors = {
        "Memory Leak": "#4c72b0",  # Soft Blue
        "Dangling Pointer": "#f28e2c",  # Soft Orange
        "Null Pointer": "#55a868",  # Soft Green
        "Use-After-Free": "#c44e52",  # Soft Red
        "Buffer Overflow": "#8172b2",  # Soft Purple
    }

    dates = []
    categories = []

    # Extract dates and categories from the filtered commits log
    with open(filtered_commits_log, "r") as f:
        for line in f:
            for category, pattern in CATEGORIES.items():
                if pattern.search(line):
                    try:
                        date = line.split()[1]
                        dates.append(datetime.strptime(date, "%Y-%m-%d"))
                        categories.append(category)
                    except (ValueError, IndexError):
                        continue

    df = pd.DataFrame({"date": dates, "category": categories})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        print("No data available for stacked distribution graph.")
        return

    # Filter for selected categories only
    df = df[df["category"].isin(selected_categories)]

    # Group by month and category
    df = df.groupby([pd.Grouper(key="date", freq="MS"), "category"]).size().unstack(fill_value=0)

    # Normalize by total number of bugs per month to get percentages
    df_percentage = df.div(df.sum(axis=1), axis=0).fillna(0) * 100

    # Reorder columns based on the selected category order
    df_percentage = df_percentage[selected_categories]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    df_percentage.plot(
        kind="area",
        stacked=True,
        ax=ax,
        color=[category_colors[cat] for cat in selected_categories],
        alpha=0.7,  # Increased transparency
    )

    # Adjust time axis and labels
    adjust_time_axis(df_percentage, ax)
    plt.title("Stacked Distribution of Bug Categories Over Time", fontsize=16)
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Percentage of Total Bugs (%)", fontsize=12)
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Bug Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stacked_bug_distribution.png")
    plt.close()

def create_bug_summary_table(results, start_date, end_date):
    """Create the bug summary table with normalized estimated man hours."""
    bugs = {
        "Memory Leak",
        "Null Pointer",
        "Use-After-Free",
        "Buffer Overflow",
        "Race Condition"
    }

    bug_summary = []
    for bug_type, count in results.items():
            # man_hours = count * bug_weights[bug_type]
            rust_feature = get_rust_language_feature(bug_type)  # Function to map Rust features to bug types
            bug_summary.append([bug_type, count, rust_feature])

    bug_summary_df = pd.DataFrame(
        bug_summary,
        columns=["Bug Type", "Count", "Rust Language Feature"]
    )
    bug_summary_df.sort_values(by="Count", ascending=False, inplace=True)
    return bug_summary_df

def get_rust_language_feature(bug_type):
    """Map Rust language features to bug types."""
    mapping = {
        "Memory Leak": "Ownership & Borrowing",
        "Null Pointer": "Option Type",
        "Use-After-Free": "Ownership & Lifetimes",
        "Buffer Overflow": "Bounds Checking",
        "Race Condition": "Thread Safety & Borrowing",
    }
    return mapping.get(bug_type, "N/A")

def save_table_as_csv(df, filename):
    """Save a DataFrame as a CSV file."""
    df.to_csv(f"{OUTPUT_DIR}/{filename}", index=False)
def render_table_as_image_with_footnote(df, title, footnote, filename):
    """Render a pandas DataFrame as a nicely formatted image with a title and footnote."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.axis('tight')

    # Add table to the plot
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Add title and footnote
    plt.title(title, fontsize=14, weight="bold")
    plt.figtext(
        0.5, 0.01, footnote, ha="center", fontsize=9, style="italic", wrap=True
    )

    # Save as an image
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def render_table_as_image(df, title, filename):
    """Render a pandas DataFrame as a table and save it as an image."""
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6))  # Dynamically scale height based on number of rows
    ax.axis("tight")
    ax.axis("off")
    
    # Create a table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust column widths dynamically
    
    # Add a title
    plt.title(title, fontsize=14, pad=20)
    
    # Save as image
    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight", dpi=300)
    plt.close()


def extract_cve_details(patches_log):
    """Extract CVE IDs and details from patch logs."""
    cve_pattern = re.compile(r"CVE-\d{4}-\d+")
    severity_keywords = {
        "High": ["critical", "severe", "privilege escalation"],
        "Medium": ["moderate", "denial of service", "dos"],
        "Low": ["information disclosure", "minor"],
    }
    vulnerability_keywords = {
        "Buffer Overflow": ["buffer overflow", "overrun"],
        "Use-After-Free": ["use-after-free", "uaf"],
        "Memory Leak": ["memory leak"],
        "Race Condition": ["race condition"],
        "Null Pointer": ["null pointer"],
    }

    cve_data = []
    for line in patches_log.splitlines():
        if "CVE" in line:
            # Extract CVE ID
            cve_ids = cve_pattern.findall(line)
            if not cve_ids:
                continue
            
            # Determine severity and type using keywords
            severity = "Unknown"
            vulnerability_type = "Unknown"
            for level, keywords in severity_keywords.items():
                if any(keyword in line.lower() for keyword in keywords):
                    severity = level
                    break
            for vtype, keywords in vulnerability_keywords.items():
                if any(keyword in line.lower() for keyword in keywords):
                    vulnerability_type = vtype
                    break
            
            # Append CVE details
            for cve_id in cve_ids:
                cve_data.append({"CVE ID": cve_id, "Severity": severity, "Type": vulnerability_type})
    return pd.DataFrame(cve_data)


def plot_cumulative_security_fixes(filtered_commits_log):
    """Plot cumulative security fixes over time for selected bug categories."""
    selected_categories = [
        "Memory Leak", "Dangling Pointer", "Null Pointer", 
        "Use-After-Free", "Buffer Overflow"
    ]
    dates = []
    categories = []

    with open(filtered_commits_log, "r") as f:
        for line in f:
            for category, pattern in CATEGORIES.items():
                if pattern.search(line):
                    try:
                        date = line.split()[1]
                        dates.append(datetime.strptime(date, "%Y-%m-%d"))
                        categories.append(category)
                    except (ValueError, IndexError):
                        continue

    df = pd.DataFrame({"date": dates, "category": categories})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if df.empty:
        print("No data available for cumulative bug fixes.")
        return

    # Filter for selected categories only
    df = df[df["category"].isin(selected_categories)]

    # Group by month and category, and compute cumulative sums
    df = df.groupby([pd.Grouper(key="date", freq="MS"), "category"]).size().unstack(fill_value=0).cumsum()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    df.plot(kind="line", ax=ax, marker="o")
    adjust_time_axis(df, ax)
    plt.title("Cumulative Security Fixes Over Time", fontsize=16)
    plt.xlabel("Time (Months)", fontsize=12)
    plt.ylabel("Cumulative Fixes", fontsize=12)
    plt.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Bug Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_bug_fixes_over_time.png")
    plt.close()
    
def main():
    print("Starting patch analysis...")

    # Step 1: Extract commits
    print("Extracting commits...")
    commits_log = extract_commits_by_keywords(REPO_PATH, KEYWORDS)

    # Step 2: Categorize commits
    print("Categorizing commits...")
    results, matches = categorize_commits(commits_log)

    # Step 3: Plot cumulative security fixes
    print("Plotting cumulative security fixes...")
    plot_cumulative_security_fixes(f"{OUTPUT_DIR}/filtered_commits.log")

    # Step 4: Plot cumulative bug trends
    print("Plotting bug trends...")
    plot_bug_trends_over_time(f"{OUTPUT_DIR}/filtered_commits.log")


    # Step 5: Plot stacked distribution
    print("Plotting stacked distribution of bug categories...")
    plot_stacked_distribution(f"{OUTPUT_DIR}/filtered_commits.log")
    

    # Summary
    print("\n--- Analysis Summary ---")
    for category, count in results.items():
        print(f"Category '{category}': {count} commits.")
    print(f"Results saved in '{OUTPUT_DIR}'.")

    # Define time range
    start_date = datetime(2022, 8, 1)
    end_date = datetime(2024, 8, 1)

    # Bug summary table
    bug_summary_table = create_bug_summary_table(results, start_date, end_date)
    print("\n--- Bug Summary Table ---")
    print(bug_summary_table)
    save_table_as_csv(bug_summary_table, "bug_summary_table.csv")
    # render_table_as_image(bug_summary_table, "Bug Summary Table", "bug_summary_table.png")
    render_table_as_image_with_footnote(
        df=bug_summary_table,
        title="Bug Summary Table",
        footnote="*estimated avg. quarterly",
        filename=f"{OUTPUT_DIR}/bug_summary_table.png",
    )

if __name__ == "__main__":
    main()
