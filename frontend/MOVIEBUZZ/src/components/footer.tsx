interface FooterProps {
  isDark?: boolean;
}

export function Footer({ isDark = true }: FooterProps) {
  const footerLinks = [
    { label: "Privacy Policy", href: "#" },
    { label: "Terms of Service", href: "#" },
    { label: "Rules of Conduct", href: "#" },
    { label: "Community Policy", href: "#" },
    { label: "Content Guidelines", href: "#" },
    { label: "Help Center", href: "#" },
    { label: "Contact Us", href: "#" },
  ];

  return (
    <footer className={`${isDark ? 'bg-black border-zinc-900' : 'bg-transparent border-zinc-300'} border-t py-8 mt-16 transition-colors duration-300`}>
      <div className="container mx-auto px-4">
        <div className="flex flex-col items-center gap-4">
          {/* Copyright */}
          <p className={`${isDark ? 'text-white' : 'text-zinc-900'} text-sm transition-colors duration-300`}>
            © 2026 MOVIEBUZZ. All rights reserved.
          </p>

          {/* Footer Links */}
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2">
            {footerLinks.map((link, index) => (
              <div key={link.label} className="flex items-center">
                <a
                  href={link.href}
                  className={`text-sm ${isDark ? 'text-[#CCCCCC] hover:text-white' : 'text-zinc-600 hover:text-zinc-900'} transition-colors`}
                  onClick={(e) => {
                    e.preventDefault();
                    console.log(`Clicked: ${link.label}`);
                  }}
                >
                  {link.label}
                </a>
                {index < footerLinks.length - 1 && (
                  <span className={`ml-4 ${isDark ? 'text-zinc-700' : 'text-zinc-300'} transition-colors duration-300`}>|</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
