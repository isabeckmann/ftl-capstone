// Worker do site: serve os arquivos estáticos de web/ e, de hora em hora (Cron Trigger), dispara o workflow
// ao_vivo.yml no GitHub. O agendador do GitHub Actions atrasa e pula rodadas (na prática, 3 ou 4 por dia);
// o do Cloudflare dispara no horário. O "schedule" do workflow continua como reserva.
//
// Secret necessário (nunca no código nem no wrangler.jsonc):
//   GITHUB_TOKEN  fine-grained PAT só deste repositório, com permissão "Actions: Read and write"
//   -> painel do Cloudflare (Worker > Settings > Variables and Secrets) ou `npx wrangler secret put GITHUB_TOKEN`

async function dispararWorkflow(env) {
  const url = `https://api.github.com/repos/${env.GITHUB_REPO}/actions/workflows/${env.GITHUB_WORKFLOW}/dispatches`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "ftl-capstone-cron",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GITHUB_REF }),
  });
  if (resp.status !== 204) {
    // Erro lançado aparece nos logs do Worker (Observability) como falha do evento agendado.
    throw new Error(`workflow_dispatch falhou: HTTP ${resp.status} ${await resp.text()}`);
  }
}

export default {
  async fetch(request, env) {
    return env.ASSETS.fetch(request);
  },

  async scheduled(controller, env, ctx) {
    if (!env.GITHUB_TOKEN) throw new Error("Secret GITHUB_TOKEN não configurado no Worker.");
    ctx.waitUntil(dispararWorkflow(env));
  },
};
