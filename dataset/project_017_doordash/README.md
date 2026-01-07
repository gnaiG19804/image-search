# DoorDash Food Delivery

## 📖 About
Leading food delivery service connecting hungry customers with their favorite local restaurants through a network of independent delivery drivers. Features real-time tracking, contactless delivery, and scheduled orders.

## 🛠 Tech Stack
- **Frontend:** React Native, React, Redux
- **Backend:** Python 3.11, Django 4.2, Celery
- **Database:** PostgreSQL, Redis, Apache Cassandra
- **DevOps:** GCP, Kubernetes, Cloud Functions

## ✨ Key Features
- [x] Real-time GPS tracking of delivery drivers
- [x] AI-powered route optimization for faster deliveries
- [x] Dynamic pricing based on demand and distance
- [x] Restaurant partner dashboard with order management
- [x] DashPass subscription for unlimited free delivery
- [x] In-app chat between customer, restaurant, and driver

## 🚀 Getting Started

### Prerequisites
- Node.js >= 18.0.0
- Python >= 3.11
- Docker and Docker Compose
- Google Maps API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/doordash/delivery-platform.git
   cd project_017_doordash
   ```

2. **Install Dependencies**
   ```bash
   # Backend
   pip install -r requirements.txt
   
   # Frontend
   cd mobile && npm install
   ```

3. **Run Development Server**
   ```bash
   docker-compose up -d
   python manage.py runserver
   ```

## 🤝 Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create.

## 📄 License
Distributed under the MIT License.
