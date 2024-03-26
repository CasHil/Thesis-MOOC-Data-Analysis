def extract_course_year(course_id: str) -> int:
    return int(course_id[-4:])

def calculate_age(course_year: int, birth_year: int) -> int:
    return course_year - birth_year

def identify_course(course_id: str) -> str:
    if 'EX101x' in course_id:
        return 'EX101x'
    elif 'ST1x' in course_id:
        return 'ST1x'
    elif 'UnixTx' in course_id:
        return 'UnixTx'
    elif 'EX101x' in course_id:
        return 'EX101x'
    else:
        return 'Other'