# Bishal's Portfolio

[![Built with Astro](https://astro.badg.es/v2/built-with-astro/small.svg)](https://astro.build)
[![Netlify Status](https://api.netlify.com/api/v1/badges/bd403085-8c0c-47f5-9c1e-a4ee44ef57bd/deploy-status)](https://app.netlify.com/sites/bishalup/deploys)

A modern, accessible portfolio showcasing AI/ML research, the Vibeset project, and professional work. Built with Astro 5, React, Tailwind CSS v4, and featuring WCAG 2.2 AA compliance.

## ✨ Features

- **Vibeset Showcase** - Interactive module presentation for the Vibeset.ai project
- **Research Section** - Dynamic research paper display with scroll-reveal animations
- **Liquid Glass Design** - Apple-inspired frosted glass effects with Google color palette
- **Magical Animations** - Framer Motion powered scroll-linked transitions
- **Dark Mode** - System-aware theme switching
- **Fully Accessible** - WCAG 2.2 AA compliant with keyboard navigation
- **SEO Optimized** - Comprehensive metadata and sitemap generation

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ (check with `node -v`)
- **npm** 9+ (check with `npm -v`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bishalu/portfolio.git
   cd portfolio
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables** (optional - for backend/AI features)
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   Navigate to [http://localhost:4321](http://localhost:4321)

## 📜 Available Commands

| Command           | Action                                       |
| :---------------- | :------------------------------------------- |
| `npm install`     | Installs dependencies                        |
| `npm run dev`     | Starts local dev server at `localhost:4321`  |
| `npm run build`   | Build your production site to `./dist/`      |
| `npm run preview` | Preview your build locally, before deploying |

## 🏗️ Tech Stack

- **Framework**: [Astro 5](https://astro.build/) with SSG/SSR hybrid
- **UI Library**: [React 19](https://react.dev/) for interactive components
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) + SCSS
- **Animations**: [Framer Motion](https://www.framer.com/motion/) + [GSAP](https://gsap.com/)
- **Icons**: [Lucide Icons](https://lucide.dev/) via `astro-icon`
- **Components**: [Accessible Astro Components](https://github.com/incluud/accessible-astro-components)
- **Deployment**: [Netlify](https://netlify.com/)

## 📁 Project Structure

```
portfolio/
├── public/              # Static assets (images, fonts, favicons)
├── src/
│   ├── assets/          # SCSS utilities and project images
│   ├── components/      # Astro & React components
│   ├── content/         # MDX content collections (projects, etc.)
│   ├── layouts/         # Page layouts
│   ├── pages/           # File-based routing
│   └── styles/          # Global Tailwind styles
├── backend/             # Netlify serverless functions
├── astro.config.mjs     # Astro configuration
├── tailwind.config.js   # Tailwind configuration
└── package.json
```

## 🔑 Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|----------|----------|-------------|
| `CONTEXT7_API_KEY` | No | Context7 API for AI documentation |
| `AWS_ID` | No | AWS access key for backend services |
| `AWS_SEC` | No | AWS secret key |
| `AWS_DEFAULT_REGION` | No | AWS region (default: `us-east-2`) |
| `MISTRAL_API_KEY` | No | Mistral AI API key |

> **Note**: The site works without any environment variables. They are only needed for optional AI/backend features.

## 🌐 Deployment

The site is configured for **Netlify** deployment:

1. Connect your GitHub repo to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variables in Netlify dashboard

Or deploy manually:
```bash
npm run build
# Upload the ./dist folder to your hosting provider
```

## 🎨 Design System

The portfolio follows a cohesive design language:

- **Google Colors**: Blue (`#4285F4`), Teal (`#00D4AA`), Moss Green (`#34A853`)
- **Apple Liquid Glass**: Frosted glass effects with backdrop blur
- **Light Organic Modern**: Soft transitions, generous whitespace
- **Typography**: Atkinson Hyperlegible + Outfit fonts

## ♿ Accessibility

- WCAG 2.2 AA compliant
- Full keyboard navigation
- Screen reader optimized
- Respects `prefers-reduced-motion`
- ARIA attributes throughout

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ by [Bishal](https://github.com/bishalu)
