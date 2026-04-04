import type { Movie } from "./api";

type SeedMetadata = {
  genres: string;
  rating: number;
};

const CURATED_METADATA: Record<string, SeedMetadata> = {
  "the-shawshank-redemption-1994": { genres: "Drama Crime", rating: 9.3 },
  "the-godfather-1972": { genres: "Crime Drama", rating: 9.2 },
  "the-dark-knight-2008": { genres: "Action Crime Drama", rating: 9.0 },
  "pulp-fiction-1994": { genres: "Crime Drama", rating: 8.9 },
  "fight-club-1999": { genres: "Drama Thriller", rating: 8.8 },
  "alien-1979": { genres: "Horror Sci-Fi", rating: 8.5 },
  "the-shining-1980": { genres: "Drama Horror", rating: 8.4 },
  "get-out-2017": { genres: "Horror Mystery Thriller", rating: 7.8 },
  "a-quiet-place-2018": { genres: "Drama Horror Sci-Fi", rating: 7.5 },
  "inception-2010": { genres: "Action Adventure Sci-Fi", rating: 8.8 },
  "interstellar-2014": { genres: "Adventure Drama Sci-Fi", rating: 8.7 },
  "the-matrix-1999": { genres: "Action Sci-Fi", rating: 8.7 },
  "forrest-gump-1994": { genres: "Drama Romance", rating: 8.8 },
  "the-lord-of-the-rings-the-fellowship-of-the-ring-2001": {
    genres: "Adventure Fantasy Action",
    rating: 8.8,
  },
  "the-lord-of-the-rings-the-two-towers-2002": {
    genres: "Adventure Fantasy Action",
    rating: 8.8,
  },
  "the-lord-of-the-rings-the-return-of-the-king-2003": {
    genres: "Adventure Fantasy Action",
    rating: 9.0,
  },
  "the-empire-strikes-back-1980": {
    genres: "Action Adventure Fantasy Sci-Fi",
    rating: 8.7,
  },
  "the-silence-of-the-lambs-1991": {
    genres: "Crime Drama Thriller",
    rating: 8.6,
  },
  "se7en-1995": { genres: "Crime Drama Thriller", rating: 8.6 },
  "gladiator-2000": { genres: "Action Adventure Drama", rating: 8.5 },
  "the-green-mile-1999": { genres: "Crime Drama Fantasy", rating: 8.6 },
  "saving-private-ryan-1998": { genres: "Drama War", rating: 8.6 },
  "the-departed-2006": { genres: "Crime Drama Thriller", rating: 8.5 },
  "whiplash-2014": { genres: "Drama Music", rating: 8.5 },
  "parasite-2019": { genres: "Drama Thriller", rating: 8.5 },
  "joker-2019": { genres: "Crime Drama Thriller", rating: 8.4 },
  "avengers-endgame-2019": { genres: "Action Adventure Sci-Fi", rating: 8.4 },
  "spider-man-2002": { genres: "Action Adventure Sci-Fi", rating: 7.4 },
  "spider-man-2-2004": { genres: "Action Adventure Sci-Fi", rating: 7.5 },
  "spider-man-homecoming-2017": { genres: "Action Adventure Sci-Fi", rating: 7.4 },
  "spider-man-into-the-spider-verse-2018": {
    genres: "Animation Action Adventure",
    rating: 8.4,
  },
  "spider-man-no-way-home-2021": {
    genres: "Action Adventure Fantasy",
    rating: 8.2,
  },
  "spider-man-across-the-spider-verse-2023": {
    genres: "Animation Action Adventure",
    rating: 8.6,
  },
  "mad-max-fury-road-2015": { genres: "Action Adventure Sci-Fi", rating: 8.1 },
  "la-la-land-2016": { genres: "Comedy Drama Romance Musical", rating: 8.0 },
  "the-prestige-2006": { genres: "Drama Mystery Sci-Fi", rating: 8.5 },
  "django-unchained-2012": { genres: "Drama Western", rating: 8.5 },
  "the-social-network-2010": { genres: "Drama Biography", rating: 7.8 },
  "blade-runner-2049-2017": { genres: "Drama Mystery Sci-Fi", rating: 8.0 },
  "the-grand-budapest-hotel-2014": {
    genres: "Comedy Adventure Crime",
    rating: 8.1,
  },
  "toy-story-1995": { genres: "Animation Adventure Comedy", rating: 8.3 },
  "toy-story-3-2010": { genres: "Animation Adventure Comedy", rating: 8.3 },
  "finding-nemo-2003": { genres: "Animation Adventure Comedy", rating: 8.2 },
  "up-2009": { genres: "Animation Adventure Comedy", rating: 8.3 },
  "coco-2017": { genres: "Animation Adventure Family", rating: 8.4 },
  "inside-out-2015": { genres: "Animation Adventure Comedy", rating: 8.1 },
  "soul-2020": { genres: "Animation Adventure Drama", rating: 8.0 },
  "moana-2016": { genres: "Animation Adventure Comedy", rating: 7.6 },
  "black-panther-2018": { genres: "Action Adventure Sci-Fi", rating: 7.3 },
  "iron-man-2008": { genres: "Action Adventure Sci-Fi", rating: 7.9 },
  "captain-america-the-winter-soldier-2014": {
    genres: "Action Adventure Sci-Fi",
    rating: 7.8,
  },
  "doctor-strange-2016": { genres: "Action Adventure Fantasy", rating: 7.5 },
  "guardians-of-the-galaxy-2014": {
    genres: "Action Adventure Comedy",
    rating: 8.0,
  },
  "top-gun-maverick-2022": { genres: "Action Drama", rating: 8.2 },
  "dune-2021": { genres: "Adventure Drama Sci-Fi", rating: 8.0 },
  "dune-part-two-2024": { genres: "Adventure Drama Sci-Fi", rating: 8.6 },
  "oppenheimer-2023": { genres: "Drama Thriller", rating: 8.3 },
  "barbie-2023": { genres: "Comedy Fantasy Adventure", rating: 6.8 },
  "the-conjuring-2013": { genres: "Horror Mystery Thriller", rating: 7.5 },
  "scream-1996": { genres: "Horror Mystery", rating: 7.4 },
  "psycho-1960": { genres: "Horror Mystery Thriller", rating: 8.5 },
  "the-exorcist-1973": { genres: "Horror", rating: 8.1 },
  "hereditary-2018": { genres: "Drama Horror Mystery", rating: 7.3 },
  "the-sixth-sense-1999": { genres: "Drama Mystery Thriller", rating: 8.2 },
};

const POSTER_PALETTES = [
  ["#111827", "#1f2937", "#475569"],
  ["#0f172a", "#1e293b", "#64748b"],
  ["#18181b", "#27272a", "#71717a"],
  ["#172033", "#24314a", "#6b7280"],
  ["#1c1917", "#292524", "#78716c"],
] as const;

function normalizeTitle(value: string) {
  return value
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function splitMovieTitle(title: string, explicitYear?: string) {
  const trimmed = title.trim();
  const match = trimmed.match(/^(.+?)\s*\((\d{4})\)\s*$/);
  if (match) {
    return {
      cleanTitle: match[1].trim(),
      year: explicitYear?.trim() || match[2],
    };
  }

  return {
    cleanTitle: trimmed,
    year: explicitYear?.trim() || "",
  };
}

function buildMovieKey(cleanTitle: string, year: string) {
  return `${normalizeTitle(cleanTitle)}-${year || "movie"}`;
}

function isMissingPoster(value?: string) {
  const poster = value?.trim().toLowerCase() || "";
  return !poster || poster.includes("placehold.co") || poster.includes("via.placeholder.com");
}

function normalizeSearchText(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/'/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function compactSearchText(value: string) {
  return normalizeSearchText(value).replace(/\s+/g, "");
}

function wrapPosterLines(text: string, maxChars = 18, maxLines = 3) {
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) {
    return ["MovieBuzz"];
  }

  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars || !current) {
      current = next;
      continue;
    }
    lines.push(current);
    current = word;
    if (lines.length === maxLines - 1) {
      break;
    }
  }

  if (lines.length < maxLines && current) {
    lines.push(current);
  }

  if (lines.length > maxLines) {
    return lines.slice(0, maxLines);
  }

  return lines;
}

export function createMoviePosterDataUrl(title: string, year: string, genres = "") {
  const seed = `${title}|${year}|${genres}`;
  const palette =
    POSTER_PALETTES[
      Array.from(seed).reduce((sum, char) => sum + char.charCodeAt(0), 0) %
        POSTER_PALETTES.length
    ];
  const [dark, primary, accent] = palette;
  const titleLines = wrapPosterLines(title);
  const genreLine = (genres || "MovieBuzz Selection").slice(0, 32);
  const titleMarkup = titleLines
    .map(
      (line, index) =>
        `<text x='28' y='${180 + index * 34}' fill='white' font-size='28' font-weight='800' font-family='Segoe UI, Arial, sans-serif'>${line}</text>`,
    )
    .join("");

  const svg = `
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 450'>
  <defs>
    <linearGradient id='bg' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0%' stop-color='${dark}' />
      <stop offset='65%' stop-color='${primary}' />
      <stop offset='100%' stop-color='${accent}' />
    </linearGradient>
    <radialGradient id='glow' cx='50%' cy='18%' r='72%'>
      <stop offset='0%' stop-color='rgba(255,255,255,0.18)' />
      <stop offset='100%' stop-color='rgba(255,255,255,0)' />
    </radialGradient>
  </defs>
  <rect width='300' height='450' rx='28' fill='url(#bg)' />
  <rect width='300' height='450' rx='28' fill='url(#glow)' />
  <rect x='18' y='18' width='264' height='414' rx='22' fill='rgba(255,255,255,0.06)' stroke='rgba(255,255,255,0.18)' />
  <rect x='28' y='118' width='244' height='5' rx='2.5' fill='rgba(248,113,113,0.78)' />
  <text x='28' y='52' fill='rgba(255,255,255,0.88)' font-size='13' font-weight='700' font-family='Segoe UI, Arial, sans-serif'>MOVIEBUZZ</text>
  <text x='28' y='92' fill='rgba(255,255,255,0.72)' font-size='12' font-weight='700' font-family='Segoe UI, Arial, sans-serif'>${genreLine}</text>
  ${titleMarkup}
  <rect x='28' y='352' width='244' height='1' fill='rgba(255,255,255,0.24)' />
  <text x='28' y='388' fill='rgba(255,255,255,0.9)' font-size='18' font-weight='700' font-family='Segoe UI, Arial, sans-serif'>${year || "Movie"}</text>
  <text x='28' y='416' fill='rgba(255,255,255,0.74)' font-size='12' font-family='Segoe UI, Arial, sans-serif'>Trailer, wishlist, and details ready</text>
</svg>
  `.trim();

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function buildFallbackDescription(movie: Movie, cleanTitle: string, year: string, genres: string) {
  if (movie.description?.trim()) {
    return movie.description.trim();
  }
  if (movie.plot?.trim()) {
    return movie.plot.trim();
  }

  const genreText = genres || "movie";
  const yearText = year ? ` from ${year}` : "";
  return `${cleanTitle} is a ${genreText.toLowerCase()} title${yearText}. Open the trailer to preview it and save it to your wishlist.`;
}

function buildTrailerLink(cleanTitle: string, year: string) {
  const search = `${cleanTitle} ${year} official trailer`.trim();
  return `https://www.youtube.com/results?search_query=${encodeURIComponent(search)}`;
}

function numericRating(value: Movie["rating"] | Movie["imdb_rating"]) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function levenshteinDistance(left: string, right: string) {
  if (!left) {
    return right.length;
  }
  if (!right) {
    return left.length;
  }

  const matrix = Array.from({ length: left.length + 1 }, () =>
    Array<number>(right.length + 1).fill(0),
  );

  for (let row = 0; row <= left.length; row += 1) {
    matrix[row][0] = row;
  }
  for (let column = 0; column <= right.length; column += 1) {
    matrix[0][column] = column;
  }

  for (let row = 1; row <= left.length; row += 1) {
    for (let column = 1; column <= right.length; column += 1) {
      const cost = left[row - 1] === right[column - 1] ? 0 : 1;
      matrix[row][column] = Math.min(
        matrix[row - 1][column] + 1,
        matrix[row][column - 1] + 1,
        matrix[row - 1][column - 1] + cost,
      );
    }
  }

  return matrix[left.length][right.length];
}

function similarityScore(left: string, right: string) {
  const longest = Math.max(left.length, right.length, 1);
  return 1 - levenshteinDistance(left, right) / longest;
}

export function enrichMovie(movie: Movie): Movie {
  const originalTitle = movie.clean_title?.trim() || movie.title.trim();
  const { cleanTitle, year } = splitMovieTitle(originalTitle || movie.title, movie.year);
  const movieKey = movie.movie_key?.trim() || buildMovieKey(cleanTitle, year);
  const seed = CURATED_METADATA[movieKey];
  const genres = movie.genres?.trim() || seed?.genres || "";
  const fallbackRating = seed?.rating;
  const rating =
    movie.rating ??
    (typeof fallbackRating === "number" ? fallbackRating : movie.imdb_rating) ??
    "N/A";
  const imdbRating =
    movie.imdb_rating?.trim() ||
    (typeof fallbackRating === "number" ? String(fallbackRating) : "");
  const poster = isMissingPoster(movie.poster)
    ? createMoviePosterDataUrl(cleanTitle, year, genres)
    : (movie.poster as string);
  const description = buildFallbackDescription(movie, cleanTitle, year, genres);

  return {
    ...movie,
    movie_key: movieKey,
    clean_title: cleanTitle,
    year,
    genres,
    poster,
    description,
    plot: movie.plot || description,
    imdb_rating: imdbRating,
    rating,
    youtube_link: movie.youtube_link || buildTrailerLink(cleanTitle, year),
  };
}

export function enrichMovies(movies: Movie[]) {
  return movies.map(enrichMovie);
}

export function movieMatchesGenre(movie: Movie, genre: string) {
  if (genre === "All") {
    return true;
  }
  return (movie.genres || "").toLowerCase().includes(genre.toLowerCase());
}

export function searchMoviesLocally(query: string, candidates: Movie[], limit = 50) {
  const normalizedQuery = normalizeSearchText(query);
  if (!normalizedQuery) {
    return [];
  }
  const compactQuery = compactSearchText(query);
  const tokens = normalizedQuery.split(" ").filter(Boolean);
  const significantTokens = tokens.filter((token) => token.length > 2);

  const rankedResults = [...candidates]
    .map((movie) => {
      const title = normalizeSearchText(`${movie.clean_title || ""} ${movie.title}`);
      const compactTitle = title.replace(/\s+/g, "");
      const description = normalizeSearchText(
        `${movie.description || ""} ${movie.plot || ""}`,
      );
      const genres = normalizeSearchText(movie.genres || "");

      let score = 0;
      if (title === normalizedQuery) {
        score += 180;
      }
      if (compactQuery && compactTitle === compactQuery) {
        score += 170;
      }
      if (title.includes(normalizedQuery)) {
        score += 125;
      }
      if (compactQuery && compactTitle.includes(compactQuery)) {
        score += 110;
      }
      if (title.startsWith(normalizedQuery)) {
        score += 45;
      }
      if (description.includes(normalizedQuery)) {
        score += 12;
      }
      if (genres.includes(normalizedQuery)) {
        score += 18;
      }

      let titleTokenHits = 0;
      let significantTitleHits = 0;
      let blobTokenHits = 0;
      for (const token of tokens) {
        const escapedToken = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const matcher = new RegExp(`\\b${escapedToken}\\b`);
        const titleMatch =
          matcher.test(title) || (token.length > 2 && compactTitle.includes(token));

        if (titleMatch) {
          score += 18;
          titleTokenHits += 1;
          if (token.length > 2) {
            significantTitleHits += 1;
          }
        }
        if (matcher.test(description) || matcher.test(genres)) {
          score += 6;
          blobTokenHits += 1;
        }
      }

      if (titleTokenHits === tokens.length && tokens.length > 0) {
        score += 70;
      } else if (tokens.length > 1) {
        score -= (tokens.length - titleTokenHits) * 52;
        if (titleTokenHits === 0) {
          score -= 24;
        }
      }

      if (significantTokens.length > 0 && significantTitleHits === significantTokens.length) {
        score += 24;
      } else if (tokens.length > 1 && significantTokens.length > 0) {
        score -= (significantTokens.length - significantTitleHits) * 36;
      }

      score += similarityScore(normalizedQuery, title) * 42;
      score += blobTokenHits * 2;
      score += numericRating(movie.rating) / 5;

      return {
        movie,
        score,
        isPreferred:
          title.includes(normalizedQuery) ||
          (compactQuery ? compactTitle.includes(compactQuery) : false) ||
          (significantTokens.length > 0
            ? significantTitleHits === significantTokens.length
            : titleTokenHits === tokens.length),
      };
    })
    .filter(({ score }) => score > 20)
    .sort((left, right) => right.score - left.score);

  const preferred = rankedResults.filter(({ isPreferred }) => isPreferred);
  const relaxed = rankedResults.filter(({ isPreferred }) => !isPreferred);

  return [...preferred, ...relaxed].slice(0, limit).map(({ movie }) => movie);
}

export function getRecommendedMovies(anchor: Movie, candidates: Movie[], limit = 50) {
  const anchorGenres = normalizeSearchText(anchor.genres || "")
    .split(" ")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  const anchorTokens = normalizeSearchText(anchor.clean_title || anchor.title)
    .split(" ")
    .map((value) => value.trim().toLowerCase())
    .filter((value) => value.length > 2);

  return [...candidates]
    .filter((movie) => movie.movie_key !== anchor.movie_key)
    .map((movie) => {
      const movieGenres = normalizeSearchText(movie.genres || "")
        .split(" ")
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean);
      const movieTokens = normalizeSearchText(movie.clean_title || movie.title)
        .split(" ")
        .map((value) => value.trim().toLowerCase())
        .filter((value) => value.length > 2);
      const sharedGenres = movieGenres.filter((genre) => anchorGenres.includes(genre)).length;
      const sharedTitleTokens = movieTokens.filter((token) =>
        anchorTokens.includes(token),
      ).length;
      const sameEra =
        anchor.year && movie.year
          ? 5 - Math.min(Math.abs(Number(anchor.year) - Number(movie.year)), 5)
          : 0;
      const score =
        sharedGenres * 20 +
        sharedTitleTokens * 10 +
        sameEra +
        numericRating(movie.rating);
      return { movie, score };
    })
    .sort((left, right) => right.score - left.score)
    .slice(0, limit)
    .map(({ movie }) => movie);
}
