import { DataAPIClient } from "@datastax/astra-db-ts";
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { Mistral } from "@mistralai/mistralai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import "dotenv/config";

const {
  ASTRA_DB_NAMESPACE,
  MISTRAL_API_KEY,
  ASTRA_DB_APPLICATION_TOKEN,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_COLLECTION,
} = process.env;

const mistral = new Mistral({
  apiKey: MISTRAL_API_KEY,
});

const f1Data = [
  "https://www.planetf1.com/news",
  "https://www.formula1.com/en/latest",
  "https://www.bbc.com/sport/formula1",
  "https://en.wikipedia.org/wiki/List_of_Formula_One_drivers",
  "https://www.skysports.com/f1https://www.skysports.com/f1",
];

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

const createCollection = async () => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION, {
    vector: {
      dimension: 1024,
      metric: "cosine",
      service: {
        provider: "mistral",
        modelName: "mistral-embed",
        authentication: {
          providerKey: MISTRAL_API_KEY,
        },
      },
    },
  });
  console.log(res);
};

const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  for await (const url of f1Data) {
    const content = await scrapePage(url);
    const chunks = await splitter.splitText(content);
    for await (const chunk of chunks) {
      const embedding = await mistral.embeddings.create({
        model: "mistral-embed",
        inputs: chunk,
        encodingFormat: "float",
      });

      const vector = embedding.data[0].embedding;
      const res = await collection.insertOne({
        $vector: vector,
        text: chunk,
      });
      console.log(res);
    }
  }
};

const scrapePage = async (url: string) => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: {
      headless: true,
    },
    gotoOptions: {
      waitUntil: "domcontentloaded",
    },
    evaluate: async (page, browser) => {
      const result = await page.evaluate(() => document.body.innerHTML);
      await browser.close();
      return result;
    },
  });
  return (await loader.scrape())?.replace(/<[^>]*>?/gm, "");
};

createCollection().then(() => loadSampleData());
