import { useState } from "react";

interface Props {
  onSearch: (q: string) => void;
  user: any;
  onLogout: () => void;
}

export default function Navbar({ onSearch, user, onLogout }: Props) {
  const [text, setText] = useState("");

  const handleChange = (e: any) => {
    setText(e.target.value);
    onSearch(e.target.value);
  };

  const username =
    user?.email?.split("@")[0] || user?.name || "User";

  return (
    <div className="navbar">
      <div className="logo">🎬 MOVIEBUZZ</div>

      <input
        className="search"
        placeholder="Search movies..."
        value={text}
        onChange={handleChange}
      />

      <div className="right">
        <span>{username}</span>
        <button onClick={onLogout}>Logout</button>
      </div>
    </div>
  );
}
