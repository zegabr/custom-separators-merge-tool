public String toString(List<T> objList) {
  if (objList.size() == 0) { return ""; }
  return objList.stream().map(obj ->
    obj.toString()).collect(Collectors.joining(","));
}
