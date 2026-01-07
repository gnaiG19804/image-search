# Etsy Marketplace Clone

## 📖 About
A vibrant online marketplace connecting independent artists, crafters, and vintage collectors with millions of buyers worldwide. Etsy specializes in handmade, vintage items, and craft supplies, offering unique products not found in traditional retail stores.

## 🛠 Tech Stack
- **Frontend:** React 18, Redux Toolkit, Material-UI
- **Backend:** Node.js, GraphQL API, Express
- **Database:** PostgreSQL 14, Elasticsearch 8, Redis
- **DevOps:** AWS ECS, Kubernetes, CloudFront CDN

## ✨ Key Features
- [x] Custom shop creation and management tools
- [x] Advanced product search with filters (price, category, location)
- [x] Seller analytics dashboard with sales insights
- [x] Integrated payment processing with multiple currencies
- [x] Direct messaging between buyers and sellers
- [x] Review and rating system for products and shops

## 🚀 Getting Started

### Prerequisites
- Node.js >= 18.0.0
- PostgreSQL >= 14.0
- Docker Desktop
- Elasticsearch cluster

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/etsy/marketplace-platform.git
   cd project_016_etsy
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Configure DATABASE_URL, ELASTICSEARCH_URL, STRIPE_API_KEY
   ```

4. **Run Development Server**
   ```bash
   npm run dev
   # Backend runs on http://localhost:4000
   # Frontend runs on http://localhost:3000
   ```

## 🏗️ Architecture
- **Microservices:** Separate services for shops, products, orders, search
- **Event-driven:** RabbitMQ for async processing of orders and notifications
- **Caching:** Redis for session management and frequently accessed data

## 🤝 Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
