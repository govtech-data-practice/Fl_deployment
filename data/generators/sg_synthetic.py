"""Synthetic Singaporean patient data generator for PSA testing.

Generates realistic multi-ethnic records (Chinese, Malay, Indian, Eurasian)
with HDB/condo addresses, and produces noisy variants with controlled
differences: romanisation variants, abbreviations, address formatting, etc.

Usage:
    from data.generators.sg_synthetic import generate_records, generate_small

    # 10,000 records with hard negatives
    clean, noisy, n_true, n_hard = generate_records(10000)

    # Small hand-crafted set for demos
    sgh, ttsh, expected = generate_small()
"""

import random
from typing import Dict, List, Tuple

# ---------------------------------------------------------------
# Name pools by ethnicity
# ---------------------------------------------------------------

CHINESE_SURNAMES = [
    "Tan", "Lim", "Lee", "Ng", "Ong", "Wong", "Goh", "Chua", "Chan", "Koh",
    "Teo", "Ang", "Yeo", "Ho", "Sim", "Chong", "Tay", "Low", "Wee", "Foo",
    "Seah", "Heng", "Poh", "Soh", "Chew", "Lau", "Yong", "Lai", "Chin", "Quek",
    "Toh", "Sng", "Pang", "Khoo", "Chia", "Gan", "Kang", "Phua", "Leong", "Loh",
]

CHINESE_MALE_NAMES = [
    "Ah Kow", "Wei Ming", "Chee Keong", "Beng Huat", "Kok Wah", "Boon Lay",
    "Wen Jie", "Jun Wei", "Zhi Hao", "Yi Xuan", "Jia Le", "Hao Yu",
    "Wei Jie", "Zhi Wei", "Jun Hao", "Jian Ming", "Kok Leong", "Boon Keng",
    "Ah Seng", "Ah Huat", "Teck Chye", "Wai Keong", "Chee Wai", "Kai Wen",
    "Yong Sheng", "De Wei", "Zheng Hao", "Rui Jie", "Jia Jun", "Hong Wei",
    "Xiang Yu", "Guo Wei", "Zi Yang", "Han Ming", "Qi Hao", "Shi Jie",
]

CHINESE_FEMALE_NAMES = [
    "Mei Ling", "Xiu Hua", "Siew Lan", "Shu Min", "Hui Ling", "Xin Yi",
    "Jia Ying", "Li Hua", "Mei Fang", "Shu Fen", "Pei Shan", "Yu Ting",
    "Hui Min", "Jia Hui", "Xin Ying", "Yi Ling", "Mei Xuan", "Shu Hui",
    "Xiu Mei", "Bao Yu", "Siew Cheng", "Bee Lian", "Ai Ling", "Sok Eng",
    "Swee Lian", "Chui Ping", "Mei Yee", "Shu Ying", "Hui Fang", "Jia Xin",
    "Li Xuan", "Wen Ting", "Zi Xin", "Yu Xin", "Xue Ting", "Rui Xin",
]

MALAY_MALE_FIRST = [
    "Muhammad", "Ahmad", "Mohamed", "Mohd", "Abdul", "Mohamad",
    "Muhamad", "Ismail", "Ibrahim", "Yusof", "Azman", "Faizal",
    "Rizal", "Hafiz", "Amir", "Firdaus", "Syafiq", "Irfan",
    "Arif", "Hakim", "Zulkifli", "Roslan", "Kamal", "Nasir",
    "Idris", "Rahman", "Shahrul", "Khairul", "Nazri", "Azhar",
]

MALAY_MALE_SECOND = [
    "Faizal", "Rizal", "Hafiz", "Syafiq", "Irfan", "Arif",
    "Hakim", "Firdaus", "Amir", "Azman", "Shahrul", "Khairul",
    "Nazri", "Azhar", "Idris", "Iskandar", "Iqbal", "Fitri",
    "Danial", "Nizam", "Haris", "Zainal", "Rashid", "Anuar",
]

MALAY_FEMALE_FIRST = [
    "Nur", "Siti", "Nurul", "Noor", "Noraini", "Faridah",
    "Rohani", "Zainab", "Fatimah", "Aminah", "Halimah", "Mariam",
    "Aisha", "Khadijah", "Safiah", "Ramlah", "Rosnah", "Suhaila",
]

MALAY_FEMALE_SECOND = [
    "Aisyah", "Nurhaliza", "Huda", "Atiqah", "Izzah", "Farhana",
    "Syahirah", "Najwa", "Aina", "Hidayah", "Zahra", "Adlina",
    "Nadiah", "Aliya", "Batrisya", "Qistina", "Insyirah", "Amira",
]

MALAY_FATHER = [
    "Abdullah", "Hassan", "Ismail", "Othman", "Ibrahim", "Yusof",
    "Ahmad", "Mohamed", "Osman", "Rahman", "Ali", "Hamid",
    "Sulaiman", "Razak", "Idris", "Kadir", "Latif", "Rahim",
    "Salleh", "Majid", "Daud", "Nordin", "Mansor", "Talib",
]

INDIAN_MALE_FIRST = [
    "Rajesh", "Suresh", "Ramesh", "Ganesh", "Prakash", "Dinesh",
    "Vikram", "Arun", "Sanjay", "Deepak", "Anand", "Mohan",
    "Kumar", "Ravi", "Vijay", "Ashok", "Gopal", "Hari",
    "Senthil", "Balakrishnan", "Murugan", "Shanmugam", "Thirumaran", "Karthik",
]

INDIAN_MALE_SECOND = [
    "Kumar", "Naidu", "Krishnan", "Rajan", "Pillai", "Nair",
    "Menon", "Iyer", "Sharma", "Singh", "Rao", "Sundaram",
]

INDIAN_FEMALE_FIRST = [
    "Priya", "Kavitha", "Lakshmi", "Devi", "Anitha", "Shalini",
    "Deepa", "Geetha", "Revathi", "Sangeetha", "Jayanthi", "Vanitha",
    "Malathi", "Pushpa", "Saraswathi", "Padma", "Meena", "Uma",
    "Nithya", "Divya", "Keerthana", "Harini", "Lavanya", "Nandhini",
]

INDIAN_FATHER = [
    "Muthu", "Rajan", "Pillai", "Krishnan", "Nair", "Menon",
    "Sundaram", "Govindan", "Perumal", "Arumugam", "Subramaniam", "Narayanan",
    "Ramasamy", "Chellappa", "Velusamy", "Thangaraj", "Maniam", "Sivalingam",
]

EURASIAN_SURNAMES = [
    "De Souza", "Pereira", "Oliveiro", "Shepherdson", "Clarke",
    "Westerhout", "Tessensohn", "Hendricks", "Sequeira", "Minjoot",
    "Scully", "Danker", "Aroozoo", "Woodford", "Aeria",
]

EURASIAN_MALE_FIRST = [
    "Daniel", "Michael", "David", "Patrick", "Adrian", "Martin",
    "Christopher", "Joseph", "Anthony", "Peter", "Raymond", "Gerald",
]

EURASIAN_FEMALE_FIRST = [
    "Michelle", "Christine", "Angela", "Jennifer", "Bernadette", "Jacqueline",
    "Yvonne", "Patricia", "Catherine", "Elizabeth", "Margaret", "Theresa",
]

# ---------------------------------------------------------------
# Address components
# ---------------------------------------------------------------

HDB_TOWNS = [
    "Ang Mo Kio Ave", "Toa Payoh Lor", "Bedok North Ave", "Jurong West St",
    "Woodlands Dr", "Tampines St", "Bukit Batok West Ave", "Yishun Ring Rd",
    "Pasir Ris St", "Hougang Ave", "Clementi Ave", "Sengkang East Way",
    "Serangoon North Ave", "Choa Chu Kang Loop", "Punggol Field",
    "Bishan St", "Queenstown Rd", "Bukit Merah View", "Marine Parade Rd",
    "Tanjong Pagar Rd", "Kallang Bahru", "Geylang East Ave",
    "Sembawang Dr", "Admiralty Dr", "Bukit Panjang Ring Rd",
]

CONDO_ROADS = [
    "Nassim Road", "Orchard Boulevard", "River Valley Rd", "Lorong Mambong",
    "Holland Rd", "Bukit Timah Rd", "East Coast Rd", "Upper Thomson Rd",
    "Jalan Bukit Merah", "Telok Blangah Rd", "Alexandra Rd", "Novena Sq",
]

# ---------------------------------------------------------------
# Noise variant tables
# ---------------------------------------------------------------

CHINESE_NAME_VARIANTS = {
    "Kow": ["Kou", "Kow"], "Keong": ["Kiong", "Keong"], "Huat": ["Hwat", "Huat"],
    "Ling": ["Ling", "Lin"], "Hua": ["Hua", "Hwa"], "Chye": ["Chai", "Chye"],
    "Wah": ["Wah", "Wa"], "Ming": ["Ming", "Min"], "Hao": ["Hao", "How"],
    "Wei": ["Wei", "Wai"], "Mei": ["Mei", "May"], "Lan": ["Lan", "Lam"],
    "Seng": ["Seng", "Sing"], "Yong": ["Yong", "Young"],
}

CHINESE_SURNAME_VARIANTS = {
    "Chen": ["Chan", "Chen"], "Wong": ["Wang", "Wong"], "Ng": ["Ng", "Wng", "Eng"],
    "Tan": ["Tan", "Tang"], "Goh": ["Goh", "Ngo"], "Chua": ["Chua", "Tsai"],
}

MALAY_PREFIXES_MALE = ["bin", "b", "B", "b.", "Bin"]
MALAY_PREFIXES_FEMALE = ["bte", "binti", "Bte", "Binti", "bt"]
MUHAMMAD_VARIANTS = ["Muhammad", "Mohamed", "Mohd", "Mohammad", "Mohamad", "Muhamad"]

INDIAN_SON_OF = ["s/o", "S/O", "s/o.", "son of"]
INDIAN_DAUGHTER_OF = ["d/o", "D/O", "d/o.", "daughter of"]

ADDRESS_ABBREVIATIONS = {
    "Blk": ["Blk", "BLK", "Block", "Blk."],
    "Ave": ["Ave", "Avenue", "Av"],
    "St": ["St", "Street", "St."],
    "Rd": ["Rd", "Road", "Rd."],
    "Dr": ["Dr", "Drive", "Dr."],
    "Lor": ["Lor", "Lorong", "Lor."],
    "North": ["North", "Nth", "N"],
    "Bukit": ["Bukit", "Bt", "Bt."],
}

# Common names for hard-negative generation
COMMON_NAMES_M = [
    "Tan Wei Ming", "Lee Jun Wei", "Lim Zhi Hao", "Ng Jun Hao",
    "Wong Jia Le", "Ong Wei Jie", "Goh Kai Wen", "Chua Zhi Wei",
    "Teo Hao Yu", "Chan Yi Xuan", "Koh Jia Jun", "Ang De Wei",
    "Muhammad Hafiz bin Abdullah", "Muhammad Irfan bin Hassan",
    "Ahmad Syafiq bin Ismail", "Mohamed Danial bin Othman",
    "Rajesh Kumar s/o Muthu", "Suresh Kumar s/o Rajan",
    "Vikram Naidu s/o Pillai", "Prakash Sharma s/o Krishnan",
]

COMMON_NAMES_F = [
    "Tan Xin Yi", "Lee Jia Ying", "Lim Hui Min", "Ng Yu Ting",
    "Wong Pei Shan", "Ong Jia Hui", "Goh Xin Ying", "Chua Yi Ling",
    "Teo Li Xuan", "Chan Wen Ting", "Koh Zi Xin", "Ang Yu Xin",
    "Nur Aisyah bte Hassan", "Nur Atiqah bte Abdullah",
    "Siti Nurhaliza bte Ibrahim", "Nurul Hidayah bte Ali",
    "Priya d/o Muthu", "Kavitha d/o Rajan",
    "Lakshmi d/o Pillai", "Deepa d/o Krishnan",
]


# ---------------------------------------------------------------
# Noise functions
# ---------------------------------------------------------------

def make_postal(postal_int: int) -> str:
    """Format postal code with random variation."""
    s = f"{postal_int:06d}"
    fmt = random.choice(["S({})", "S{}", "Singapore {}", "S ({})"])
    return fmt.format(s)


def add_name_noise(name: str, is_chinese_given: bool = False) -> str:
    """Add subtle noise to a name part."""
    if is_chinese_given:
        parts = name.split()
        noised = []
        for p in parts:
            if p in CHINESE_NAME_VARIANTS:
                noised.append(random.choice(CHINESE_NAME_VARIANTS[p]))
            else:
                noised.append(p)
        if len(noised) == 2 and random.random() < 0.15:
            return "-".join(noised)
        return " ".join(noised)
    return name


def add_address_noise(address: str) -> str:
    """Add formatting noise to an address string."""
    result = address
    for canonical, variants in ADDRESS_ABBREVIATIONS.items():
        if canonical in result:
            replacement = random.choice(variants)
            result = result.replace(canonical, replacement, 1)
    return result


def random_hdb_address() -> str:
    """Generate a random HDB address."""
    blk = random.randint(100, 999)
    town = random.choice(HDB_TOWNS)
    town_num = random.randint(1, 15)
    floor = random.randint(1, 25)
    unit = random.randint(1, 999)
    postal = random.randint(100000, 829999)
    return f"Blk {blk} {town} {town_num} #{floor:02d}-{unit:03d} S({postal:06d})"


def random_condo_address() -> str:
    """Generate a random condo address."""
    road = random.choice(CONDO_ROADS)
    num = random.randint(1, 200)
    floor = random.randint(1, 30)
    unit = random.randint(1, 20)
    postal = random.randint(100000, 829999)
    return f"{num} {road} #{floor:02d}-{unit:02d} S({postal:06d})"


# ---------------------------------------------------------------
# Small hand-crafted dataset (for demos / tutorials)
# ---------------------------------------------------------------

def generate_small():
    """Generate a small hand-crafted dataset of 20 Singaporean patients.

    Returns:
        (sgh_records, ttsh_records, expected_matches)
    """
    sgh = [
        {"name": "Tan Ah Kow",         "dob": "1965-08-12", "address": "Blk 123 Ang Mo Kio Ave 6 #08-456 S(560123)",  "gender": "M"},
        {"name": "Lim Mei Ling",       "dob": "1978-03-22", "address": "Blk 456 Toa Payoh Lor 1 #12-789 S(310456)",   "gender": "F"},
        {"name": "Wong Wei Ming",      "dob": "1982-11-05", "address": "Blk 789 Bedok North Ave 4 #03-123 S(460789)", "gender": "M"},
        {"name": "Chen Xiu Hua",       "dob": "1970-06-30", "address": "Blk 234 Jurong West St 42 #05-678 S(640234)", "gender": "F"},
        {"name": "Ng Chee Keong",      "dob": "1958-01-15", "address": "Blk 567 Woodlands Dr 14 #11-234 S(730567)",   "gender": "M"},
        {"name": "Lee Siew Lan",       "dob": "1985-09-18", "address": "Blk 890 Tampines St 81 #07-345 S(520890)",    "gender": "F"},
        {"name": "Ong Beng Huat",      "dob": "1973-12-25", "address": "Blk 345 Bukit Batok West Ave 8 #09-012 S(650345)", "gender": "M"},
        {"name": "Goh Shu Min",        "dob": "1990-04-08", "address": "Blk 678 Yishun Ring Rd #02-567 S(760678)",    "gender": "F"},
        {"name": "Muhammad Faizal bin Abdullah",   "dob": "1975-07-14", "address": "Blk 111 Pasir Ris St 12 #06-333 S(510111)", "gender": "M"},
        {"name": "Nur Aisyah bte Hassan",          "dob": "1988-02-28", "address": "Blk 222 Hougang Ave 10 #10-444 S(530222)", "gender": "F"},
        {"name": "Ahmad Rizal bin Ismail",         "dob": "1969-10-03", "address": "Blk 333 Clementi Ave 2 #04-555 S(120333)", "gender": "M"},
        {"name": "Siti Nurhaliza bte Othman",      "dob": "1992-05-17", "address": "Blk 444 Sengkang East Way #08-666 S(540444)", "gender": "F"},
        {"name": "Rajesh Kumar s/o Muthu",     "dob": "1967-04-22", "address": "Blk 555 Serangoon North Ave 1 #03-777 S(550555)", "gender": "M"},
        {"name": "Priya Lakshmi d/o Rajan",    "dob": "1983-08-10", "address": "Blk 666 Choa Chu Kang Loop #11-888 S(680666)",   "gender": "F"},
        {"name": "Suresh Naidu s/o Pillai",    "dob": "1971-12-01", "address": "Blk 777 Punggol Field #05-999 S(820777)",        "gender": "M"},
        {"name": "Kavitha Devi d/o Krishnan",  "dob": "1979-06-15", "address": "Blk 888 Bishan St 13 #09-111 S(570888)",         "gender": "F"},
        {"name": "Daniel De Souza",    "dob": "1986-03-07", "address": "12 Nassim Road #04-05 S(258370)",              "gender": "M"},
        {"name": "Michelle Pereira",   "dob": "1994-11-20", "address": "45 Lorong Mambong S(277694)",                  "gender": "F"},
        {"name": "Koh Boon Lay",       "dob": "1960-02-14", "address": "Blk 999 Pioneer Rd #01-222 S(640999)",         "gender": "M"},
        {"name": "Hamidah bte Yusof",  "dob": "1991-07-07", "address": "Blk 100 Admiralty Dr #06-100 S(730100)",       "gender": "F"},
    ]

    ttsh = [
        {"name": "Tan Ah Kou",          "dob": "1965-08-12", "address": "BLK 123 Ang Mo Kio Ave 6 #08-456 S560123",    "gender": "M"},
        {"name": "Lim Mei-Ling",        "dob": "1978-03-22", "address": "Blk 456 Toa Payoh Lorong 1 #12-789 S(310456)","gender": "F"},
        {"name": "Wong Wei Ming",       "dob": "1982-11-05", "address": "Blk 789 Bedok Nth Ave 4 #03-123 S(460789)",   "gender": "M"},
        {"name": "Chan Xiu Hua",        "dob": "1970-06-30", "address": "Blk 234 Jurong West St 42 #05-678 S640234",   "gender": "F"},
        {"name": "Ng Chee Kiong",       "dob": "1958-01-15", "address": "Blk 567 Woodlands Drive 14 #11-234 S(730567)","gender": "M"},
        {"name": "Lee Siew Lan",        "dob": "1985-09-18", "address": "Blk 890 Tampines Street 81 #07-345 S(520890)","gender": "F"},
        {"name": "Ong Beng Hwat",       "dob": "1973-12-25", "address": "BLK 345 Bt Batok West Ave 8 #09-012 S(650345)","gender": "M"},
        {"name": "Goh Shu Min",         "dob": "1990-04-08", "address": "Blk 678 Yishun Ring Road #02-567 S(760678)",  "gender": "F"},
        {"name": "Mohd Faizal b Abdullah",         "dob": "1975-07-14", "address": "Blk 111 Pasir Ris St 12 #06-333 S(510111)", "gender": "M"},
        {"name": "Nur Aisyah binti Hassan",        "dob": "1988-02-28", "address": "Blk 222 Hougang Ave 10 #10-444 S530222",   "gender": "F"},
        {"name": "Ahmad Rizal B Ismail",           "dob": "1969-10-03", "address": "BLK 333 Clementi Avenue 2 #04-555 S(120333)", "gender": "M"},
        {"name": "Siti Nurhaliza bte Osman",       "dob": "1992-05-17", "address": "Blk 444 Sengkang East Way #08-666 S(540444)", "gender": "F"},
        {"name": "Rajesh Kumar S/O Muthu",     "dob": "1967-04-22", "address": "Blk 555 Serangoon Nth Ave 1 #03-777 S(550555)", "gender": "M"},
        {"name": "Priya Lakshmi D/O Rajan",    "dob": "1983-08-10", "address": "Blk 666 Choa Chu Kang Loop #11-888 S680666",   "gender": "F"},
        {"name": "Suresh Naidu s/o Pillai",    "dob": "1971-12-01", "address": "Blk 777 Punggol Field #05-999 S(820777)",      "gender": "M"},
        {"name": "Kavita Devi d/o Krishnan",   "dob": "1979-06-15", "address": "Blk 888 Bishan Street 13 #09-111 S(570888)",   "gender": "F"},
        {"name": "Daniel de Souza",    "dob": "1986-03-07", "address": "12 Nassim Rd #04-05 S(258370)",               "gender": "M"},
        {"name": "Michelle Pereira",   "dob": "1994-11-20", "address": "45 Lorong Mambong Singapore 277694",          "gender": "F"},
        {"name": "Teo Kok Wah",       "dob": "1955-09-30", "address": "Blk 200 Novena Sq #03-200 S(308079)",          "gender": "M"},
        {"name": "Nurul Huda bte Ali", "dob": "1993-04-12", "address": "Blk 300 Kallang Bahru #05-300 S(330300)",      "gender": "F"},
    ]

    return sgh, ttsh, 18  # first 18 are true matches


# ---------------------------------------------------------------
# Large-scale generator
# ---------------------------------------------------------------

def _generate_record(eth: str, gender: str, age: int, income: int):
    """Generate a single clean record and its noisy variant."""
    dob = f"{2026 - age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    address = random_hdb_address() if random.random() < 0.85 else random_condo_address()

    if eth == "chinese":
        surname = random.choice(CHINESE_SURNAMES)
        given = random.choice(CHINESE_MALE_NAMES if gender == "M" else CHINESE_FEMALE_NAMES)
        name = f"{surname} {given}"
        noisy_surname = surname
        if surname in CHINESE_SURNAME_VARIANTS and random.random() < 0.1:
            noisy_surname = random.choice(CHINESE_SURNAME_VARIANTS[surname])
        noisy_name = f"{noisy_surname} {add_name_noise(given, is_chinese_given=True)}"

    elif eth == "malay":
        if gender == "M":
            first = random.choice(MALAY_MALE_FIRST)
            second = random.choice(MALAY_MALE_SECOND)
            father = random.choice(MALAY_FATHER)
            name = f"{first} {second} bin {father}"
            noisy_first = random.choice(MUHAMMAD_VARIANTS) if first in MUHAMMAD_VARIANTS else first
            noisy_father = "Osman" if father == "Othman" and random.random() < 0.4 else father
            noisy_name = f"{noisy_first} {second} {random.choice(MALAY_PREFIXES_MALE)} {noisy_father}"
        else:
            first = random.choice(MALAY_FEMALE_FIRST)
            second = random.choice(MALAY_FEMALE_SECOND)
            father = random.choice(MALAY_FATHER)
            name = f"{first} {second} bte {father}"
            noisy_father = "Osman" if father == "Othman" and random.random() < 0.4 else father
            noisy_name = f"{first} {second} {random.choice(MALAY_PREFIXES_FEMALE)} {noisy_father}"

    elif eth == "indian":
        if gender == "M":
            first = random.choice(INDIAN_MALE_FIRST)
            second = random.choice(INDIAN_MALE_SECOND)
            father = random.choice(INDIAN_FATHER)
            name = f"{first} {second} s/o {father}"
            noisy_name = f"{first} {second} {random.choice(INDIAN_SON_OF)} {father}"
        else:
            first = random.choice(INDIAN_FEMALE_FIRST)
            father = random.choice(INDIAN_FATHER)
            name = f"{first} d/o {father}"
            noisy_first = first
            if first == "Kavitha" and random.random() < 0.3: noisy_first = "Kavita"
            elif first == "Lakshmi" and random.random() < 0.3: noisy_first = "Laxmi"
            elif first == "Anitha" and random.random() < 0.3: noisy_first = "Anita"
            noisy_name = f"{noisy_first} {random.choice(INDIAN_DAUGHTER_OF)} {father}"

    else:  # eurasian
        surname = random.choice(EURASIAN_SURNAMES)
        first = random.choice(EURASIAN_MALE_FIRST if gender == "M" else EURASIAN_FEMALE_FIRST)
        name = f"{first} {surname}"
        noisy_surname = surname.replace("De ", "de ") if "De " in surname and random.random() < 0.3 else surname
        noisy_name = f"{first} {noisy_surname}"

    # Noisy address
    noisy_address = add_address_noise(address)
    if "S(" in noisy_address:
        try:
            postal_str = noisy_address.split("S(")[-1].rstrip(")")
            postal_int = int(postal_str)
            noisy_address = noisy_address[:noisy_address.index("S(")] + make_postal(postal_int)
        except ValueError:
            pass

    # Noisy age/income
    noisy_age = age + random.choice([-1, 1]) if random.random() < 0.05 else age
    noisy_income = int(round(income / 500) * 500) if random.random() < 0.1 else income

    clean = {"name": name, "dob": dob, "address": address,
             "gender": gender, "age": str(age), "income": str(income)}
    noisy = {"name": noisy_name, "dob": dob, "address": noisy_address,
             "gender": gender, "age": str(noisy_age), "income": str(noisy_income)}
    return clean, noisy


def generate_records(n: int = 10000, seed: int = 42, hard_negatives: bool = True):
    """Generate n Singaporean patient records with noisy variants and hard negatives.

    Args:
        n: Number of true-match record pairs to generate.
        seed: Random seed for reproducibility.
        hard_negatives: If True, append adversarial records (same-name-diff-person,
                        family members, neighbours, movers).

    Returns:
        (clean_records, noisy_records, n_true_matches, n_hard_negatives)
    """
    random.seed(seed)
    ethnic_weights = [0.74, 0.13, 0.09, 0.04]
    ethnicities = ["chinese", "malay", "indian", "eurasian"]

    clean, noisy = [], []

    for _ in range(n):
        eth = random.choices(ethnicities, weights=ethnic_weights, k=1)[0]
        gender = random.choice(["M", "F"])
        age = random.randint(18, 95)
        age_factor = max(0.3, 1.0 - abs(age - 45) / 40)
        income = max(800, min(25000, int(round(random.gauss(4500, 2500) * age_factor / 100) * 100)))

        rec_clean, rec_noisy = _generate_record(eth, gender, age, income)
        clean.append(rec_clean)
        noisy.append(rec_noisy)

    n_hard = 0
    if not hard_negatives:
        return clean, noisy, n, 0

    # Type 1: Same name, different person (10%)
    n_same_name = n // 10
    for _ in range(n_same_name):
        gender = random.choice(["M", "F"])
        shared_name = random.choice(COMMON_NAMES_M if gender == "M" else COMMON_NAMES_F)
        age_a, age_b = random.randint(18, 95), random.randint(18, 95)
        while abs(age_b - age_a) < 3:
            age_b = random.randint(18, 95)
        noisy_name = shared_name
        for old, pool in [("bin", MALAY_PREFIXES_MALE), ("bte", MALAY_PREFIXES_FEMALE),
                          ("s/o", INDIAN_SON_OF), ("d/o", INDIAN_DAUGHTER_OF)]:
            if old in shared_name:
                noisy_name = shared_name.replace(old, random.choice(pool))
                break
        clean.append({"name": shared_name, "dob": f"{2026-age_a}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                      "address": random_hdb_address(), "gender": gender, "age": str(age_a), "income": str(random.randint(800, 25000))})
        noisy.append({"name": noisy_name, "dob": f"{2026-age_b}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                      "address": add_address_noise(random_hdb_address()), "gender": gender, "age": str(age_b), "income": str(random.randint(800, 25000))})

    # Type 2: Family members — same surname + address, different given name (5%)
    n_families = n // 20
    for _ in range(n_families):
        surname = random.choice(CHINESE_SURNAMES)
        base_addr = random_hdb_address()
        father_given = random.choice(CHINESE_MALE_NAMES)
        son_given = random.choice(CHINESE_MALE_NAMES)
        while son_given == father_given:
            son_given = random.choice(CHINESE_MALE_NAMES)
        father_age = random.randint(45, 75)
        son_age = random.randint(18, father_age - 20)
        clean.append({"name": f"{surname} {father_given}", "dob": f"{2026-father_age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                      "address": base_addr, "gender": "M", "age": str(father_age), "income": str(random.randint(3000, 15000))})
        noisy.append({"name": f"{surname} {son_given}", "dob": f"{2026-son_age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                      "address": add_address_noise(base_addr), "gender": "M", "age": str(son_age), "income": str(random.randint(800, 8000))})

    # Type 3: Neighbours — same block, different unit (5%)
    n_neighbours = n // 20
    for _ in range(n_neighbours):
        blk = random.randint(100, 999)
        town = random.choice(HDB_TOWNS)
        tn = random.randint(1, 15)
        fl = random.randint(1, 25)
        u_a = random.randint(1, 998)
        postal = random.randint(100000, 829999)
        addr_a = f"Blk {blk} {town} {tn} #{fl:02d}-{u_a:03d} S({postal:06d})"
        addr_b = f"Blk {blk} {town} {tn} #{fl:02d}-{u_a+1:03d} S({postal:06d})"
        for g, addr in [("M", addr_a), ("F", addr_b)]:
            pass
        age_a, age_b = random.randint(18, 95), random.randint(18, 95)
        g_a, g_b = random.choice(["M", "F"]), random.choice(["M", "F"])
        clean.append({"name": f"{random.choice(CHINESE_SURNAMES)} {random.choice(CHINESE_MALE_NAMES if g_a == 'M' else CHINESE_FEMALE_NAMES)}",
                      "dob": f"{2026-age_a}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "address": addr_a,
                      "gender": g_a, "age": str(age_a), "income": str(random.randint(800, 25000))})
        noisy.append({"name": f"{random.choice(CHINESE_SURNAMES)} {random.choice(CHINESE_MALE_NAMES if g_b == 'M' else CHINESE_FEMALE_NAMES)}",
                      "dob": f"{2026-age_b}-{random.randint(1,12):02d}-{random.randint(1,28):02d}", "address": add_address_noise(addr_b),
                      "gender": g_b, "age": str(age_b), "income": str(random.randint(800, 25000))})

    # Type 4: Movers — same name + DOB, completely different address (2%)
    n_movers = n // 50
    for _ in range(n_movers):
        gender = random.choice(["M", "F"])
        surname = random.choice(CHINESE_SURNAMES)
        given = random.choice(CHINESE_MALE_NAMES if gender == "M" else CHINESE_FEMALE_NAMES)
        name = f"{surname} {given}"
        age = random.randint(18, 95)
        income = random.randint(800, 25000)
        dob = f"{2026-age}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        clean.append({"name": name, "dob": dob, "address": random_hdb_address(),
                      "gender": gender, "age": str(age), "income": str(income)})
        noisy.append({"name": name, "dob": dob, "address": add_address_noise(random_hdb_address()),
                      "gender": gender, "age": str(age), "income": str(income)})

    n_hard = n_same_name + n_families + n_neighbours + n_movers
    return clean, noisy, n, n_hard
