/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
        './src/components/**/*.{js,ts,jsx,tsx,mdx}',
        './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                background: 'var(--background)',
                foreground: 'var(--foreground)',
                'bg-secondary': 'var(--bg-secondary)',
                'text-secondary': 'var(--text-secondary)',
                border: 'var(--border)',
                accent: 'var(--accent)',
                'card-data': 'var(--card-data)',
                'card-model': 'var(--card-model)',
                'card-evaluation': 'var(--card-evaluation)',
                'card-inference': 'var(--card-inference)',
                'status-pending': 'var(--status-pending)',
                'status-running': 'var(--status-running)',
                'status-completed': 'var(--status-completed)',
                'status-failed': 'var(--status-failed)',
            },
        },
    },
    plugins: [],
    darkMode: 'class',
};
