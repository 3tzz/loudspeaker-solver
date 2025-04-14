import requests
from bs4 import BeautifulSoup

# Step 1: Fetch the page content
url = "https://loudspeakerdatabase.com/PRV/6MB400"  # Replace with the actual URL of the site
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Parse the table data
# Assuming the data is in a table with <tr> rows and <td> columns

data = []
for row in soup.find_all("tr")[1:]:  # Skip the header row
    columns = row.find_all("td")
    if len(columns) >= 3:  # Ensure the row has enough columns
        frequency = columns[0].get_text().strip()
        impedance = columns[1].get_text().strip()
        phase = columns[2].get_text().strip()

        # Add to the data list
        data.append({"frequency": frequency, "impedance": impedance, "phase": phase})

# Step 3: Print the scraped data
for entry in data:
    print(
        f"Frequency: {entry['frequency']} Hz, Impedance: {entry['impedance']} Ω, Phase: {entry['phase']}°"
    )
