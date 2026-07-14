-- Force every table to have explicit, equal column widths.
--
-- Why this exists: pandoc only emits wrapping `p{width}` columns for a LaTeX
-- table when it decides the table is "wide". For our 4-column comparison tables
-- (Criterion | B1 | B2 | B3, each cell a paragraph of prose) it guessed wrong,
-- emitted non-wrapping `llll` columns, and the table ran straight off the right
-- edge of the page, silently truncating the content. A table that loses its own
-- data is worse than no table.
--
-- Setting the widths explicitly makes the LaTeX writer use `p{}` columns, which
-- wrap. Column count is preserved; nothing else is touched.

function Table(tbl)
  local ncols = #tbl.colspecs
  if ncols == 0 then return nil end

  local widths_missing = false
  for _, spec in ipairs(tbl.colspecs) do
    -- spec is {alignment, width}; width is nil or 'ColWidthDefault' when unset
    if spec[2] == nil or spec[2] == 'ColWidthDefault' then
      widths_missing = true
    end
  end
  if not widths_missing then return nil end

  local w = 1.0 / ncols
  for i = 1, ncols do
    tbl.colspecs[i][2] = w
  end
  return tbl
end
