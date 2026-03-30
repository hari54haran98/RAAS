"""
DAY 26: JWT Authentication Module
Handles user registration, login, and token validation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    tier: str = "free"  # free/premium/admin


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload data."""
    username: Optional[str] = None
    tier: Optional[str] = None


class AuthManager:
    """Manages authentication and authorization."""

    def __init__(self):
        print("=" * 60)
        print("DAY 26: JWT AUTHENTICATION")
        print("=" * 60)

        # In-memory user database (replace with real DB in production)
        self.users_db = {}

        # Create default admin user
        self.create_user(
            username="admin",
            password="admin123",
            email="admin@raas.com",
            full_name="Admin User",
            tier="admin"
        )

        print(f"✓ JWT configured (algorithm: {ALGORITHM})")
        print(f"✓ Token expiry: {ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
        print("✓ Default admin user created (change password in production)")
        print("=" * 60)

    def get_password_hash(self, password: str) -> str:
        """Hash a password using bcrypt directly."""
        # bcrypt works on bytes, and has a 72 byte limit
        password_bytes = password.encode('utf-8')[:72]
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hash using bcrypt directly."""
        try:
            password_bytes = plain_password.encode('utf-8')[:72]
            hashed_bytes = hashed_password.encode('utf-8')
            return bcrypt.checkpw(password_bytes, hashed_bytes)
        except Exception:
            return False

    def get_user(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self.users_db.get(username)

    def create_user(self, username: str, password: str, email: str,
                    full_name: Optional[str] = None, tier: str = "free") -> UserInDB:
        """Create a new user."""
        hashed_password = self.get_password_hash(password)

        user = UserInDB(
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            tier=tier
        )
        self.users_db[username] = user
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify a JWT token and return token data."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            tier: str = payload.get("tier", "free")
            if username is None:
                return None
            return TokenData(username=username, tier=tier)
        except JWTError:
            return None

    def get_current_user(self, token: str) -> Optional[UserInDB]:
        """Get current user from token."""
        token_data = self.verify_token(token)
        if token_data is None:
            return None
        user = self.get_user(token_data.username)
        return user


# Quick test
if __name__ == "__main__":
    auth = AuthManager()

    print("\n🧪 TESTING AUTHENTICATION")
    print("-" * 40)

    # Create test user
    auth.create_user("testuser", "testpass123", "test@example.com", "Test User")
    print("✓ Created test user")

    # Test authentication
    user = auth.authenticate_user("testuser", "testpass123")
    if user:
        print(f"✓ Authentication successful: {user.username}")
    else:
        print("❌ Authentication failed")

    # Test token creation
    if user:
        token = auth.create_access_token({"sub": user.username, "tier": user.tier})
        print(f"✓ Token created: {token[:30]}...")

        # Test token verification
        token_data = auth.verify_token(token)
        if token_data:
            print(f"✓ Token verified: user={token_data.username}, tier={token_data.tier}")
        else:
            print("❌ Token verification failed")