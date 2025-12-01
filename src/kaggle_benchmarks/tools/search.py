# Copyright 2025 Kaggle Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kaggle_benchmarks import actors, chats, utils


class SearchBackendError(RuntimeError):
    pass


class SearchBackendBase:
    def __init__(self):
        self.name = "Search"
        self.avatar = "🔎"

    def __call__(self, search_query: str) -> str:
        raise NotImplementedError


class SearchBackendWikipedia(SearchBackendBase):
    def __init__(self, language="en"):
        self.name = "Wikipedia"
        self.avatar = "📖"
        self.language = language
        self.client = utils.build_httpx_client("wikipedia")

    def search_wikipedia_articles(
        self, search_query: str, number_of_results=10
    ) -> list[dict[str, str]]:
        """Searches for Wikipedia articles based on a query.

        Args:
            search_query: The query to search for.
            number_of_results: The maximal number of results to return.

        Returns:
            A list of dictionaries, each representing a Wikipedia page.
        """
        url = f"https://api.wikimedia.org/core/v1/wikipedia/{self.language}/search/page"
        parameters = {"q": search_query, "limit": number_of_results}

        response = self.client.get(url, headers={}, params=parameters)
        hits = response.json()["pages"]
        keys = ("key", "title", "description")
        return [{k: entry[k] for k in keys} for entry in hits]

    def get_wikipedia_artricle_by_key(self, page_key: str) -> str:
        """Retrieves the content of a Wikipedia article given its page key.

        Args:
            page_key: The page key of the Wikipedia article.

        Returns:
            The content of the Wikipedia article as a string.
        """
        base_url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "titles": page_key,
            "format": "json",
            "explaintext": True,  # Get plain text, no HTML
        }

        response = self.client.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if pages:
            page = next(iter(pages.values()))  # Get the first (and likely only) page
            return page.get("extract", "")

        raise SearchBackendError(f"No results for key `{page_key}`")

    def __call__(self, search_query: str) -> str:
        """I'm Feeling Lucky type search on Wikipedia.

        This search always returns the first article found on Wikipedia for a
        given query.

        Args:
            search_query: Search query used to find the article.

        Returns:
            The content of the first article found on Wikipedia.
        """

        pages = self.search_wikipedia_articles(search_query, number_of_results=1)
        if len(pages) == 0:
            raise SearchBackendError(f"No results for query `{search_query}`")

        page_key = pages[0]["key"]
        return self.get_wikipedia_artricle_by_key(page_key)


_AVAILABLE_BACKENDS: dict[str, SearchBackendBase] = {
    "wikipedia": SearchBackendWikipedia(),
}


class SearchEngine(actors.Actor):
    def __init__(self, backend: str | SearchBackendBase):
        if isinstance(backend, str):
            backend = backend.lower()
            if backend not in _AVAILABLE_BACKENDS:
                raise SearchBackendError(f"Backend `{backend}` not recognized!")

            self.backend = _AVAILABLE_BACKENDS[backend]
        else:
            self.backend = backend

        assert isinstance(self.backend, SearchBackendBase)

        super().__init__(
            name=self.backend.name,
            role="tool",
            avatar=self.backend.avatar,
        )

    def respond(self) -> str:
        chat = chats.get_current_chat()
        return self.search(chat.messages[-1].text)

    @chats.emits_message
    def search(self, message: str) -> str:
        return self.backend(message)
