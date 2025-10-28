
-- Verificando IDs do Sitema Legado do TCU

from docs
select key, id:try_cast(string_split(key, '-')[-1] as bigint)
where key ilike '%legada%' and id is not null
order by id asc;

-- Verificando IDs do Sitema Atual do TCU

from docs
select key, id:try_cast(string_split(key, '-')[-1] as bigint)
where key not ilike '%legada%' and id is not null
order by id asc;