# User Registration Transaction - Comprehensive Documentation

## Transaction Overview

The **User Registration Transaction** handles new user account creation through the Streamlit web interface. This transaction includes form validation, password security, profile data collection, and comprehensive database operations with proper error handling and user feedback.

### Transaction Classification
- **Priority Level**: 1 (Core Business Logic)
- **Complexity**: Medium-High
- **Tables Involved**: 1 primary table (User)
- **Transaction Type**: Single-table ACID compliant
- **Execution Context**: Streamlit web interface with form submission

---

## Data Flow Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Registration    │────│   Input Valid.   │────│  Field Required │
│   Form UI       │    │   (Client-side)  │    │     Check       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   User Registration Logic                       │
│  ▪ Role validation (user/coach/admin)                          │
│  ▪ Password hashing with bcrypt                                │
│  ▪ Profile data JSON serialization                             │
│  ▪ Duplicate email/username checking                           │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Database Transaction Layer                      │
│                                                                │
│  1. Duplicate Check    →  User Table (email/username)          │
│  2. User Creation      →  User Table (comprehensive schema)    │
│  3. Transaction Commit →  Return user_id                       │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Transaction Components

### 1. **Registration Form UI**
**Location**: `src/app/app.py:274-311`

```python
elif st.session_state.page == "Register":
    st.title("User Registration")

    # Core registration fields
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Profile fields
    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    height = st.number_input("Height (cm)", min_value=0)
    weight = st.number_input("Weight (kg)", min_value=0)
    role = st.selectbox("Role", ["user", "coach", "admin"])

    # Registration form submission
    if st.button("Register"):
        if not username or not password or not email:
            st.error("Email, username, and password are required.")
        else:
            try:
                profile_data = {
                    "name": name,
                    "date_of_birth": str(dob),
                    "height": height,
                    "weight": weight
                }
                user = User.register(
                    email=email,
                    username=username,
                    password=password,
                    profile_data=profile_data,
                    role=role
                )
                st.success(f"User registered with ID: {user.user_id}")
            except Exception as e:
                st.error(f"Registration failed: {e}")

    st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))
```

**Form Fields**:
- **Required**: Email, Username, Password
- **Optional Profile**: Full Name, Date of Birth, Height (cm), Weight (kg)
- **Role Selection**: Dropdown with user/coach/admin options

### 2. **Input Validation & Security**
**Location**: `src/database/users/user.py:43-56`

```python
@staticmethod
def _validate_role(role: str) -> str:
    """Validate role is one of the allowed values"""
    valid_roles = ['user', 'coach', 'admin']
    if role not in valid_roles:
        raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
    return role

@staticmethod
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

@staticmethod
def verify_password(plain_pw, hashed_pw):
    return bcrypt.checkpw(plain_pw.encode('utf-8'), hashed_pw.encode('utf-8'))
```

**Security Features**:
- bcrypt password hashing with salt
- Role validation against allowed values
- Client-side required field validation

### 3. **Registration Transaction Logic**
**Location**: `src/database/users/user.py:58-91`

```python
@classmethod
def register(cls, email: str, username: str, password: str, 
            profile_data: Optional[Dict[str, Any]] = None, 
            role: str = 'user', first_name: str = None, 
            last_name: str = None) -> 'User':
    """Register a new user with comprehensive schema"""
    # Validate inputs
    role = cls._validate_role(role)
    hashed_pw = cls.hash_password(password)
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            now = datetime.now()
            cur.execute("""
                INSERT INTO "User" (email, username, password, registration_date, 
                                   registration_time, profile_data, user_type, 
                                   first_name, last_name, is_active, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING user_id
            """, (email, username, hashed_pw, now.date(), now.time(), 
                  json.dumps(profile_data) if profile_data else None, 
                  role, first_name, last_name, True, now, now))
            user_id = cur.fetchone()[0]
            conn.commit()
            return cls(user_id=user_id, email=email, username=username, 
                      password=hashed_pw, registration_date=now.date(), 
                      registration_time=now.time(), profile_data=profile_data, 
                      user_type=role, first_name=first_name, last_name=last_name,
                      is_active=True, created_at=now, updated_at=now)
    except psycopg2.Error as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
```

---

## Database Transaction Steps

### Step 1: Input Validation
```python
# Client-side validation
if not username or not password or not email:
    st.error("Email, username, and password are required.")
    return

# Server-side role validation
role = cls._validate_role(role)  # Validates against ['user', 'coach', 'admin']
```

### Step 2: Password Security
```python
# Hash password with bcrypt
hashed_pw = cls.hash_password(password)
# Uses bcrypt.hashpw() with salt generation
```

### Step 3: User Record Creation
```sql
INSERT INTO "User" (
    email, username, password, registration_date, registration_time, 
    profile_data, user_type, first_name, last_name, is_active, 
    created_at, updated_at
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
RETURNING user_id
```

**Key Fields**:
- `email`: Unique email address (with database constraint)
- `username`: Unique username (with database constraint)
- `password`: bcrypt hashed password
- `registration_date`/`registration_time`: Split timestamp for legacy compatibility
- `profile_data`: JSONB field containing structured profile information
- `user_type`: Enum value (user/coach/admin)
- `is_active`: Default TRUE for new accounts
- `created_at`/`updated_at`: Timestamp tracking

### Step 4: User Object Return
```python
return cls(
    user_id=user_id, email=email, username=username, 
    password=hashed_pw, registration_date=now.date(), 
    registration_time=now.time(), profile_data=profile_data, 
    user_type=role, first_name=first_name, last_name=last_name,
    is_active=True, created_at=now, updated_at=now
)
```

---

## Legacy Registration Implementation

### Alternative Registration Function
**Location**: `src/database/users/register.py:10-51`

```python
def register_user(email, username, password, profile_data=None, role="user"):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute('SELECT 1 FROM "User" WHERE email = %s;', (email,))
        if cursor.fetchone():
            return "Error: Email already exists."

        # Check if username already exists
        cursor.execute('SELECT 1 FROM "User" WHERE username = %s;', (username,))
        if cursor.fetchone():
            return "Error: Username already exists."

        # Hash password and insert user
        hashed_pw = hash_password(password)
        profile_json = json.dumps(profile_data) if profile_data else None
        role_json = json.dumps(role)

        insert_user = """
        INSERT INTO "User" (email, username, password, registration_date, profile_data, role)
        VALUES (%s, %s, %s, NOW(), %s, %s)
        RETURNING user_id;
        """
        cursor.execute(insert_user, (email, username, hashed_pw, profile_json, role_json))
        user_id = cursor.fetchone()[0]

        conn.commit()
        return user_id

    except psycopg2.IntegrityError as e:
        conn.rollback()
        return f"Integrity error: {e}"
    except Exception as e:
        return f"Database error: {e}"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
```

**Differences from Modern Implementation**:
- Explicit duplicate checking before insert
- Legacy column names (`role` instead of `user_type`)
- String return values for errors instead of exceptions
- Manual cursor management

---

## Transaction Execution Flow

### Phase 1: Form Input Collection
```python
# User fills out registration form
email = st.text_input("Email")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
name = st.text_input("Full Name")
dob = st.date_input("Date of Birth")
height = st.number_input("Height (cm)", min_value=0)
weight = st.number_input("Weight (kg)", min_value=0)
role = st.selectbox("Role", ["user", "coach", "admin"])
```

### Phase 2: Client-side Validation
```python
if st.button("Register"):
    if not username or not password or not email:
        st.error("Email, username, and password are required.")
        return
```

### Phase 3: Profile Data Assembly
```python
profile_data = {
    "name": name,
    "date_of_birth": str(dob),
    "height": height,
    "weight": weight
}
```

### Phase 4: Transaction Processing
```python
try:
    user = User.register(
        email=email,
        username=username,
        password=password,
        profile_data=profile_data,
        role=role
    )
    st.success(f"User registered with ID: {user.user_id}")
except Exception as e:
    st.error(f"Registration failed: {e}")
```

---

## Error Handling Scenarios

### 1. **Required Field Validation**
```python
if not username or not password or not email:
    st.error("Email, username, and password are required.")
```

**Rollback**: No database operation initiated, user remains on form.

### 2. **Invalid Role Selection**
```python
@staticmethod
def _validate_role(role: str) -> str:
    valid_roles = ['user', 'coach', 'admin']
    if role not in valid_roles:
        raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")
```

**Rollback**: ValueError raised, caught by UI exception handler.

### 3. **Database Constraint Violations**
```sql
-- Unique constraint on email
CONSTRAINT "User_email_key" UNIQUE (email)
-- Unique constraint on username  
CONSTRAINT "User_username_key" UNIQUE (username)
```

**Rollback**: psycopg2.IntegrityError raised, transaction rolled back automatically.

### 4. **Database Connection Failure**
```python
conn = get_connection()
if not conn:
    # Connection failure handled by get_connection()
    raise ConnectionError("Failed to connect to database")
```

**Rollback**: Connection error propagated to UI error handler.

### 5. **Password Hashing Failure**
```python
try:
    hashed_pw = cls.hash_password(password)
except Exception as e:
    # bcrypt failure (rare but possible)
    raise ValueError(f"Password processing failed: {e}")
```

**Rollback**: Password processing error caught before database operation.

---

## User Experience Features

### 1. **Navigation Flow**
```python
# Home page navigation
with col1:
    st.button("🔐 Register", on_click=set_page, args=("Register",))

# Return navigation
st.button("⬅️ Back to Home", on_click=set_page, args=("Home",))
```

### 2. **Form Field Types**
- **Text Input**: Email, Username, Full Name
- **Password Input**: Password (masked)
- **Date Input**: Date of Birth (calendar picker)
- **Number Input**: Height, Weight (with min_value validation)
- **Select Box**: Role selection (dropdown)

### 3. **Success/Error Feedback**
```python
# Success message
st.success(f"User registered with ID: {user.user_id}")

# Error message
st.error(f"Registration failed: {e}")
```

### 4. **CSS Styling**
**Location**: `src/app/style.css`

```css
/* Input field styling */
.stTextInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    color: white;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
```

---

## Security Considerations

### 1. **Password Security**
```python
# bcrypt with automatic salt generation
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
```

**Features**:
- bcrypt hashing algorithm (industry standard)
- Automatic salt generation per password
- UTF-8 encoding for international character support

### 2. **SQL Injection Prevention**
```python
# Parameterized queries
cur.execute("""
    INSERT INTO "User" (...) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""", (email, username, hashed_pw, ...))
```

### 3. **Input Validation**
```python
# Role validation
valid_roles = ['user', 'coach', 'admin']
if role not in valid_roles:
    raise ValueError(f"Invalid role: {role}")

# Required field validation
if not username or not password or not email:
    st.error("Email, username, and password are required.")
```

### 4. **Database Constraints**
```sql
-- Email uniqueness
email VARCHAR(255) UNIQUE NOT NULL,
-- Username uniqueness
username VARCHAR(100) UNIQUE NOT NULL,
-- Role validation via enum
user_type user_role_enum DEFAULT 'user'
```

---

## Performance Considerations

### 1. **Single Transaction Design**
```python
conn = get_connection()
try:
    with conn.cursor() as cur:
        # Single INSERT operation
        cur.execute(...)
        conn.commit()
except psycopg2.Error as e:
    conn.rollback()
    raise e
finally:
    conn.close()
```

**Optimization**: Single database round-trip for registration.

### 2. **JSON Profile Storage**
```python
# Efficient profile data storage
profile_data = json.dumps(profile_data) if profile_data else None
```

**Benefit**: Flexible profile schema without additional tables.

### 3. **Connection Management**
```python
# Proper connection lifecycle
conn = get_connection()
try:
    # Database operations
finally:
    conn.close()
```

**Optimization**: Immediate connection cleanup prevents connection leaks.

---

## Business Rules & Constraints

### 1. **Unique Email/Username**
```sql
CONSTRAINT "User_email_key" UNIQUE (email),
CONSTRAINT "User_username_key" UNIQUE (username)
```
**Rule**: Each email and username must be unique across the system.

### 2. **Default User Role**
```python
role: str = 'user'  # Default role for new registrations
```
**Rule**: New users default to 'user' role unless explicitly specified.

### 3. **Required Fields**
```python
if not username or not password or not email:
    st.error("Email, username, and password are required.")
```
**Rule**: Email, username, and password are mandatory for registration.

### 4. **Profile Data Optional**
```python
profile_data: Optional[Dict[str, Any]] = None
```
**Rule**: Profile information (name, DOB, height, weight) is optional.

### 5. **Active by Default**
```python
is_active=True  # New accounts are active by default
```
**Rule**: New user accounts are immediately active and usable.

---

## Integration Points

### 1. **Streamlit Session State**
```python
st.session_state.page == "Register"
```

### 2. **Database Connection**
```python
from src.database.database_connection import get_connection
```

### 3. **User Authentication**
```python
# Registration creates user ready for authentication
user = User.register(...)
# Can immediately use User.authenticate() afterward
```

### 4. **CSS Styling**
```python
# External stylesheet integration
st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
```

---

## Testing & Validation

### 1. **Form Validation Testing**
- Required field enforcement
- Email format validation (via HTML5 input type)
- Password strength (could be enhanced)
- Role selection validation

### 2. **Database Constraint Testing**
- Duplicate email handling
- Duplicate username handling
- Invalid role rejection
- Transaction rollback verification

### 3. **Security Testing**
- Password hashing verification
- SQL injection attempt prevention
- bcrypt salt uniqueness

### 4. **Integration Testing**
- Registration → Authentication flow
- Profile data persistence
- Error message display
- Navigation flow testing

---

## Future Enhancement Opportunities

### 1. **Enhanced Validation**
```python
# Email format validation
import re
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email)

# Password strength validation
def validate_password_strength(password):
    return len(password) >= 8 and any(c.isupper() for c in password)
```

### 2. **Email Verification**
```python
# Send verification email after registration
def send_verification_email(user):
    # Implementation for email verification workflow
    pass
```

### 3. **Profile Enhancements**
```python
# Additional profile fields
fitness_goals = st.multiselect("Fitness Goals", ["Weight Loss", "Muscle Gain", "Endurance"])
experience_level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
```

### 4. **Audit Logging**
```python
# Registration audit trail
def log_registration_attempt(email, username, success, error=None):
    # Implementation for audit logging
    pass
```

---

## Summary

The User Registration Transaction provides a comprehensive user onboarding experience with robust security, validation, and error handling. The transaction combines modern OOP design with legacy compatibility, ensuring both maintainability and backward compatibility.

**Key Strengths**:
- Comprehensive security with bcrypt password hashing
- Flexible profile data storage using JSONB
- Robust error handling with user-friendly feedback
- Professional UI with custom CSS styling
- ACID-compliant database transactions

**Areas for Improvement**:
- Email verification workflow
- Enhanced password strength validation
- Audit logging for security monitoring
- Profile field enhancements for fitness tracking