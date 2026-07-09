import random
from typing import Dict, List

GENDERS: List[str] = ["male", "female", "non-binary"]

MALE_FIRST_NAMES: List[str] = [
    "Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas",
    "Henry", "Alexander", "Mason", "Michael", "Ethan", "Daniel", "Jacob",
    "Logan", "Jackson", "Sebastian", "Jack", "Aiden", "Owen", "Samuel",
    "Matthew", "Joseph", "Levi", "Mateo", "David", "John", "Wyatt", "Carter",
    "Julian", "Luke", "Grayson", "Isaac", "Jayden", "Theodore", "Caleb",
    "Ryan", "Nathan", "Adrian", "Christian", "Thomas", "Elias", "Aaron",
    "Charles", "Evan", "Jonathan", "Dominic", "Luca", "Hunter", "Austin",
    "Robert", "Andrew", "Hudson", "Allan", "Victor", "Santiago", "Micheal",
    "Tobias", "Zachary", "Riley", "Bryce", "Silas", "Emmett", "Gideon",
    "Marcus", "Nicolas", "Eddie", "Finley", "Spencer", "Kian", "Lorenzo",
    "Kameron", "Jasper", "Heath", "Craig", "Quentin", "Harvey", "Callum",
    "Zane", "Ari", "Angelo", "Ronan", "Kellan", "Darren", "Frederick", "Kurt",
]

FEMALE_FIRST_NAMES: List[str] = [
    "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Charlotte",
    "Amelia", "Harper", "Evelyn", "Abigail", "Emily", "Ella", "Elizabeth",
    "Camila", "Luna", "Sofia", "Avery", "Mila", "Aria", "Scarlett", "Penelope",
    "Layla", "Chloe", "Victoria", "Madison", "Eleanor", "Grace", "Nora",
    "Riley", "Zoey", "Hannah", "Hazel", "Lily", "Ellie", "Violet", "Lillian",
    "Aurora", "Natalie", "Emilia", "Stella", "Zoe", "Leah", "Audrey", "Claire",
    "Skylar", "Lucy", "Paisley", "Anna", "Caroline", "Nova", "Genesis",
    "Alice", "Madeline", "Cora", "Bella", "Vivian", "Kennedy", "Maya",
    "Willow", "Kinsley", "Naomi", "Elena", "Sarah", "Aaliyah", "Allison",
    "Gabriella", "Sadie", "Arianna", "Kaylee", "Serenity", "Hailey", "Mackenzie",
    "Gianna", "Charlie", "Valentina", "Nicole", "Julie", "Reagan", "Elise",
    "Jade", "Mila", "Lydia", "Jillian", "Ophelia", "Rebecca", "Stacy",
]

NON_BINARY_FIRST_NAMES: List[str] = [
    "Alex", "Taylor", "Jordan", "Casey", "Riley", "Quinn", "Morgan", "Avery",
    "Reese", "Blake", "Cameron", "Drew", "Harper", "Finley", "Peyton",
    "Rowan", "Sage", "Sydney", "Emerson", "Hayden", "Jesse", "Kai", "Logan",
    "Mackenzie", "Parker", "Reagan", "Sawyer", "Tatum", "Zion", "Elliot",
    "Keegan", "Skyler", "Dakota", "Pat", "River", "Shiloh", "Winter",
    "Phoenix", "Robin", "Kelly", "Kendall", "Dayton", "Lennon", "Milan",
    "Sinclair", "Arden", "Briar", "Jules", "Marley", "Sloane", "Ellis",
]

# --- Last names (global, gender‑neutral) -------------------------------
LAST_NAMES: List[str] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos",
    "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez",
    "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long",
    "Ross", "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell",
    "Sullivan", "Bell", "Coleman", "Butler", "Henderson", "Barnes",
    "Gonzales", "Fisher", "Vasquez", "Simmons", "Romero", "Jordan",
    "Patterson", "Alexander", "Hamilton", "Graham", "Reynolds", "Griffin"
]

OCCUPATIONS: List[str] = [
    "Software Engineer", "Data Scientist", "Product Manager", "Graphic Designer",
    "Digital Marketer", "UX/UI Designer", "Financial Analyst",
    "Human Resources Specialist", "Operations Manager", "Content Writer",
    "Project Coordinator", "Quality Assurance Engineer", "Customer Success Manager",
    "Business Analyst", "Systems Administrator", "Network Engineer",
    "Research Scientist", "Accountant", "Legal Counsel", "Logistics Coordinator",
    "Supply Chain Analyst", "Medical Assistant", "Pharmacist",
    "Physical Therapist", "Registered Nurse", "Veterinarian",
    "Teacher", "University Lecturer", "Curriculum Developer",
    "Event Planner", "Public Relations Specialist", "Social Media Manager",
    "Copywriter", "Technical Writer", "Video Editor", "Animator",
    "Game Developer", "Cybersecurity Analyst", "Cloud Architect",
    "DevOps Engineer", "Data Engineer", "Machine Learning Engineer",
    "AI Researcher", "Robotics Engineer", "Astronomer", "Environmental Scientist",
    "Geologist", "Archaeologist", "Urban Planner", "Economist",
    "Policy Analyst", "Consultant", "Sales Representative", "Real Estate Agent",
    "Chef", "Nutritionist", "Barista", "Flight Attendant", "Pilot",
    "Marine Biologist", "Park Ranger", "Conservationist", "Writer",
    "Poet", "Photographer", "Journalist", "Editor", "Translator",
    "Interpreter", "Social Worker", "Psychologist", "Counselor",
    "Therapist", "Dentist", "Optometrist", "Dermatologist",
    "Surgeon", "Radiologist", "Medical Researcher", "Lab Technician",
    "Biochemist", "Chemist", "Physicist", "Mathematician",
    "Statistician", "Economist", "Logistics Manager", "Warehouse Supervisor",
    "Construction Manager", "Architect", "Interior Designer", "Landscape Architect",
    "Electrician", "Plumber", "Carpenter", "Mechanic", "Automotive Engineer",
    "Industrial Designer", "Fashion Designer", "Makeup Artist", "Hair Stylist",
    "Personal Trainer", "Yoga Instructor", "Life Coach", "Mediator",
    "Freelancer", "Entrepreneur", "Startup Founder", "Investor",
    "Venture Capitalist", "Non‑profit Director", "Volunteer Coordinator"
]

def _pick_name(gender: str) -> str:
    """Return a gender‑appropriate first name."""
    if gender == "male":
        return random.choice(MALE_FIRST_NAMES)
    if gender == "female":
        return random.choice(FEMALE_FIRST_NAMES)
    # non‑binary – mix all three pools for maximum variety
    all_names = MALE_FIRST_NAMES + FEMALE_FIRST_NAMES + NON_BINARY_FIRST_NAMES
    return random.choice(all_names)


def _pick_age(min_age: int = 18, max_age: int = 80) -> int:
    """Return a random age within a realistic adult range."""
    return random.randint(min_age, max_age)


def generate_random_persona(*, gender, age_range = (18, 80)):
    chosen_gender = gender if gender in GENDERS else random.choice(GENDERS)

    first_name = _pick_name(chosen_gender)
    last_name = random.choice(LAST_NAMES)

    min_age, max_age = age_range
    age = _pick_age(min_age, max_age)

    occupation = random.choice(OCCUPATIONS)

    persona: Dict[str, str | int] = {
        "gender": chosen_gender,
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "occupation": occupation,
    }

    return persona